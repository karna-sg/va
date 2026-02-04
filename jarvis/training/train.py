"""
LoRA Fine-Tuning for Jarvis Voice Assistant (Tier 2 Router)

Fine-tunes Qwen3-4B via MLX LoRA on personalized intent classification data.
Trains the local LLM to better route Vasu's voice commands.

Pipeline:
  1. Load seed + augmented training data
  2. Format as chat-template instruction/response pairs
  3. Run LoRA fine-tuning via mlx_lm
  4. Save adapter weights
  5. Optionally merge adapter into base model

Usage:
  python -m jarvis.training.train [--epochs 3] [--lr 1e-5] [--output adapters/]
  python -m jarvis.training.train --merge  # Merge adapter into base

Performance: ~500 iterations, ~10 min on Apple Silicon M4.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any


# Training data format for mlx_lm LoRA
CHAT_TEMPLATE = """<|im_start|>system
Intent classifier for Jarvis, a voice assistant for developer Vasu. Output ONLY a single-line JSON object.

Intents: github.list_issues(repo,state), github.get_issue(repo,number), github.create_issue(repo,title,body), github.list_prs(repo,state), github.get_pr(repo,number), github.list_commits(repo,since), github.activity_yesterday(repo), github.activity_today(repo), github.activity_this_week(repo), github.repo_status(repo), slack.post_message(channel,message), slack.list_channels, slack.read_messages(channel), git.status, git.diff, git.branch, code.implement(needs_claude=true), code.fix_bug(needs_claude=true), code.review(needs_claude=true), code.explain(needs_claude=true), code.refactor(needs_claude=true), cli.run_tests, cli.run_build, workflow.daily_status(channel), workflow.pr_review(repo,number), workflow.sprint_planning(repo), meta.greeting, meta.thanks, meta.help, meta.cancel.

Format: {{"intent":"x","confidence":0.9,"params":{{}},"needs_claude":false,"missing_slots":[]}}
Unknown: {{"intent":"unknown","confidence":0.0,"params":{{}},"needs_claude":true,"missing_slots":[]}}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
<think>

</think>

{output}<|im_end|>"""


def prepare_training_data(output_dir: str = "~/.jarvis/training") -> str:
    """
    Prepare training data in mlx_lm LoRA format.

    Creates train.jsonl and valid.jsonl in the output directory.

    Returns:
        Path to the output directory
    """
    from jarvis.training.data import get_seed_samples
    from jarvis.training.augment import augment_dataset

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get and augment seed data
    seed = get_seed_samples()
    print("Seed samples: %d" % len(seed))

    augmented = augment_dataset(seed, target_multiplier=3.5)
    print("Augmented samples: %d" % len(augmented))

    # Shuffle and split: 90% train, 10% validation
    import random
    random.seed(42)
    random.shuffle(augmented)

    split_idx = int(len(augmented) * 0.9)
    train_samples = augmented[:split_idx]
    valid_samples = augmented[split_idx:]

    # Format for mlx_lm LoRA (chat template format)
    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")

    _write_formatted(train_samples, train_path)
    _write_formatted(valid_samples, valid_path)

    print("Train: %d samples -> %s" % (len(train_samples), train_path))
    print("Valid: %d samples -> %s" % (len(valid_samples), valid_path))

    return output_dir


def _write_formatted(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Write samples in mlx_lm chat format (JSONL with 'text' field)"""
    with open(output_path, 'w') as f:
        for sample in samples:
            output_json = json.dumps({
                "intent": sample["intent"],
                "confidence": 0.95,
                "params": sample["params"],
                "needs_claude": sample["needs_claude"],
                "missing_slots": sample["missing_slots"],
            })

            text = CHAT_TEMPLATE.format(
                input=sample["utterance"],
                output=output_json,
            )

            f.write(json.dumps({"text": text}) + "\n")


def train_lora(
    data_dir: str = "~/.jarvis/training",
    model_name: str = "Qwen/Qwen3-4B-MLX-4bit",
    adapter_dir: str = "~/.jarvis/adapters",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    lora_rank: int = 8,
    lora_layers: int = 16,
    steps_per_report: int = 10,
    steps_per_eval: int = 50,
    max_seq_length: int = 512,
) -> str:
    """
    Run LoRA fine-tuning on the local Qwen3-4B model.

    Args:
        data_dir: Directory with train.jsonl and valid.jsonl
        model_name: Base model to fine-tune
        adapter_dir: Where to save adapter weights
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for LoRA
        lora_rank: LoRA rank (lower = fewer params, faster)
        lora_layers: Number of layers to apply LoRA to
        steps_per_report: Print loss every N steps
        steps_per_eval: Run validation every N steps
        max_seq_length: Maximum sequence length

    Returns:
        Path to saved adapter directory
    """
    data_dir = os.path.expanduser(data_dir)
    adapter_dir = os.path.expanduser(adapter_dir)
    os.makedirs(adapter_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl")

    if not os.path.exists(train_path):
        print("Training data not found. Run prepare_training_data() first.")
        return ""

    # Count samples
    with open(train_path) as f:
        train_count = sum(1 for _ in f)
    with open(valid_path) as f:
        valid_count = sum(1 for _ in f)

    total_steps = (train_count * epochs) // batch_size
    print("Training: %d samples x %d epochs = %d steps" % (
        train_count, epochs, total_steps))
    print("Validation: %d samples" % valid_count)
    print("LoRA: rank=%d, layers=%d, lr=%.1e" % (lora_rank, lora_layers, learning_rate))
    print()

    # Run mlx_lm LoRA training via CLI
    # This is the most reliable way to invoke mlx_lm training
    import subprocess

    # Write LoRA config YAML if non-default rank
    config_path = None
    if lora_rank != 8:
        import yaml
        config_path = os.path.join(data_dir, "lora_config.yaml")
        lora_config = {
            "lora_parameters": {"rank": lora_rank, "dropout": 0.0, "scale": 20.0},
        }
        with open(config_path, 'w') as cf:
            yaml.dump(lora_config, cf)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_name,
        "--data", data_dir,
        "--adapter-path", adapter_dir,
        "--train",
        "--batch-size", str(batch_size),
        "--num-layers", str(lora_layers),
        "--iters", str(total_steps),
        "--learning-rate", str(learning_rate),
        "--steps-per-report", str(steps_per_report),
        "--steps-per-eval", str(steps_per_eval),
        "--max-seq-length", str(max_seq_length),
    ]

    if config_path:
        cmd.extend(["--config", config_path])

    print("Running: %s" % " ".join(cmd))
    print()

    start = time.time()

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
    )

    duration = time.time() - start
    print("\nTraining completed in %.1f minutes" % (duration / 60))

    if result.returncode != 0:
        print("Training failed with exit code %d" % result.returncode)
        return ""

    print("Adapter saved to: %s" % adapter_dir)
    return adapter_dir


def merge_adapter(
    model_name: str = "Qwen/Qwen3-4B-MLX-4bit",
    adapter_dir: str = "~/.jarvis/adapters",
    output_dir: str = "~/.jarvis/merged_model",
) -> str:
    """
    Merge LoRA adapter into the base model for faster inference.

    The merged model can be loaded directly without adapter overhead.
    """
    adapter_dir = os.path.expanduser(adapter_dir)
    output_dir = os.path.expanduser(output_dir)

    import subprocess

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_name,
        "--adapter-path", adapter_dir,
        "--save-path", output_dir,
    ]

    print("Merging adapter into base model...")
    print("Running: %s" % " ".join(cmd))

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print("Merge failed with exit code %d" % result.returncode)
        return ""

    print("Merged model saved to: %s" % output_dir)
    return output_dir


def main():
    """CLI for LoRA fine-tuning"""
    import argparse

    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for Jarvis voice assistant")
    parser.add_argument("--prepare", action="store_true",
                        help="Prepare training data only")
    parser.add_argument("--train", action="store_true",
                        help="Run LoRA fine-tuning")
    parser.add_argument("--merge", action="store_true",
                        help="Merge adapter into base model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--layers", type=int, default=16,
                        help="LoRA layers (default: 16)")
    parser.add_argument("--data-dir", default="~/.jarvis/training",
                        help="Training data directory")
    parser.add_argument("--adapter-dir", default="~/.jarvis/adapters",
                        help="Adapter output directory")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-MLX-4bit",
                        help="Base model name")

    args = parser.parse_args()

    if args.prepare or args.train:
        print("=" * 50)
        print("  Preparing training data...")
        print("=" * 50)
        data_dir = prepare_training_data(args.data_dir)

    if args.train:
        print()
        print("=" * 50)
        print("  Starting LoRA fine-tuning...")
        print("=" * 50)
        adapter_dir = train_lora(
            data_dir=args.data_dir,
            model_name=args.model,
            adapter_dir=args.adapter_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            lora_rank=args.rank,
            lora_layers=args.layers,
        )

    if args.merge:
        print()
        print("=" * 50)
        print("  Merging adapter into base model...")
        print("=" * 50)
        merge_adapter(
            model_name=args.model,
            adapter_dir=args.adapter_dir,
        )

    if not (args.prepare or args.train or args.merge):
        parser.print_help()
        print("\nExamples:")
        print("  python -m jarvis.training.train --prepare         # Prepare data only")
        print("  python -m jarvis.training.train --train           # Prepare + train")
        print("  python -m jarvis.training.train --train --epochs 5 --lr 2e-5")
        print("  python -m jarvis.training.train --merge           # Merge adapter")


if __name__ == "__main__":
    main()
