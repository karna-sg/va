"""
Routing Accuracy Evaluation for Kat Voice Assistant

Compares the fine-tuned model vs base model on intent classification:
1. Load test utterances with known intents
2. Run both base and fine-tuned models
3. Report accuracy, latency, and per-intent breakdown

Usage:
  python -m jarvis.training.evaluate                    # Test base model
  python -m jarvis.training.evaluate --adapter ~/.kat/adapters  # Test fine-tuned
  python -m jarvis.training.evaluate --compare          # A/B comparison
"""

import json
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Result of evaluating a single utterance"""
    utterance: str
    expected_intent: str
    predicted_intent: str
    correct: bool
    confidence: float
    duration_ms: float
    params_match: bool


def get_test_cases() -> List[Dict[str, Any]]:
    """
    Get test cases for evaluation.

    These are held-out samples NOT in the training data,
    representing realistic voice commands for Vasu's workflow.
    """
    return [
        # GitHub Issues
        {"utterance": "show me what bugs are open", "intent": "github.list_issues"},
        {"utterance": "any tickets on the board", "intent": "github.list_issues"},
        {"utterance": "tell me about issue 25", "intent": "github.get_issue"},
        {"utterance": "what is ticket number 8", "intent": "github.get_issue"},
        {"utterance": "file a bug for the search crash", "intent": "github.create_issue"},

        # GitHub PRs
        {"utterance": "any PRs waiting for me", "intent": "github.list_prs"},
        {"utterance": "show pull request 15", "intent": "github.get_pr"},
        {"utterance": "what's in PR 3", "intent": "github.get_pr"},

        # GitHub Activity
        {"utterance": "what did we ship yesterday", "intent": "github.activity_yesterday"},
        {"utterance": "any progress today", "intent": "github.activity_today"},
        {"utterance": "week in review for curiescious", "intent": "github.activity_this_week"},
        {"utterance": "show me the commit log", "intent": "github.list_commits"},
        {"utterance": "how is curiescious looking", "intent": "github.repo_status"},

        # Slack
        {"utterance": "message the engineering team", "intent": "slack.post_message"},
        {"utterance": "post this to slack please", "intent": "slack.post_message"},
        {"utterance": "what channels are there", "intent": "slack.list_channels"},
        {"utterance": "check slack for me", "intent": "slack.read_messages"},

        # Git
        {"utterance": "anything uncommitted", "intent": "git.status"},
        {"utterance": "show me what changed", "intent": "git.diff"},
        {"utterance": "which branch am I on", "intent": "git.branch"},

        # Code (needs Claude)
        {"utterance": "code up the notification system", "intent": "code.implement"},
        {"utterance": "debug the login failure", "intent": "code.fix_bug"},
        {"utterance": "look over the recent changes", "intent": "code.review"},
        {"utterance": "how does the payment flow work", "intent": "code.explain"},
        {"utterance": "tidy up the API module", "intent": "code.refactor"},

        # CLI
        {"utterance": "make sure the tests pass", "intent": "cli.run_tests"},
        {"utterance": "compile everything", "intent": "cli.run_build"},

        # Workflows
        {"utterance": "give the team a status update", "intent": "workflow.daily_status"},
        {"utterance": "do a thorough review of PR 5", "intent": "workflow.pr_review"},
        {"utterance": "what should we prioritize for the sprint", "intent": "workflow.sprint_planning"},

        # Meta
        {"utterance": "hey kat good morning", "intent": "meta.greeting"},
        {"utterance": "cheers kat", "intent": "meta.thanks"},
        {"utterance": "what are you capable of", "intent": "meta.help"},
        {"utterance": "forget about it", "intent": "meta.cancel"},

        # Unknown / edge cases
        {"utterance": "what is the weather like", "intent": "unknown"},
        {"utterance": "tell me about quantum computing", "intent": "unknown"},
    ]


def evaluate_model(model_name: str = "Qwen/Qwen3-4B-MLX-4bit",
                    adapter_path: Optional[str] = None) -> List[EvalResult]:
    """
    Evaluate a model (base or fine-tuned) on the test suite.

    Args:
        model_name: Base model name
        adapter_path: Path to LoRA adapter (None = base model only)

    Returns:
        List of EvalResult for each test case
    """
    from jarvis.llm.local_router import LocalRouter

    # Load model
    if adapter_path:
        print("Loading fine-tuned model (adapter: %s)..." % adapter_path)
        router = LocalRouter(model_name=model_name)
        # For adapter loading, we need to load with adapter
        _load_with_adapter(router, model_name, adapter_path)
    else:
        print("Loading base model: %s..." % model_name)
        router = LocalRouter(model_name=model_name)
        router.initialize()

    if not router.is_ready:
        print("Model failed to load!")
        return []

    test_cases = get_test_cases()
    results = []

    print("\nRunning %d test cases...\n" % len(test_cases))

    for tc in test_cases:
        result = router.route(tc["utterance"])

        if result:
            predicted = result.intent
            correct = predicted == tc["intent"]
            eval_r = EvalResult(
                utterance=tc["utterance"],
                expected_intent=tc["intent"],
                predicted_intent=predicted,
                correct=correct,
                confidence=result.confidence,
                duration_ms=result.duration_ms,
                params_match=True,  # simplified
            )
        else:
            eval_r = EvalResult(
                utterance=tc["utterance"],
                expected_intent=tc["intent"],
                predicted_intent="PARSE_FAIL",
                correct=False,
                confidence=0.0,
                duration_ms=0.0,
                params_match=False,
            )

        results.append(eval_r)

        flag = "OK" if eval_r.correct else "XX"
        print("%s %5.0fms  %-45s  expected=%-25s got=%s" % (
            flag, eval_r.duration_ms,
            eval_r.utterance[:45],
            eval_r.expected_intent,
            eval_r.predicted_intent,
        ))

    return results


def _load_with_adapter(router, model_name: str, adapter_path: str) -> None:
    """Load model with LoRA adapter applied"""
    try:
        from mlx_lm import load

        print("  Loading model with LoRA adapter...")
        print("  Base: %s" % model_name)
        print("  Adapter: %s" % adapter_path)
        model, tokenizer = load(model_name, adapter_path=adapter_path)

        router._model = model
        router._tokenizer = tokenizer
        router._warm_system_cache()
        router._is_ready = True
        print("  Fine-tuned model ready")

    except Exception as e:
        print("  Failed to load adapter: %s" % e)
        print("  Falling back to base model...")
        router.initialize()


def print_report(results: List[EvalResult], label: str = "Model") -> Dict[str, Any]:
    """Print an evaluation report"""
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    parse_fails = sum(1 for r in results if r.predicted_intent == "PARSE_FAIL")
    times = [r.duration_ms for r in results if r.duration_ms > 0]

    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = (sum(times) / len(times)) if times else 0

    print("\n" + "=" * 60)
    print("  %s Evaluation Report" % label)
    print("=" * 60)
    print("  Accuracy: %d/%d (%.1f%%)" % (correct, total, accuracy))
    print("  Parse failures: %d" % parse_fails)
    if times:
        print("  Latency: avg=%.0fms min=%.0fms max=%.0fms" % (
            avg_time, min(times), max(times)))

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.expected_intent.split('.')[0] if '.' in r.expected_intent else r.expected_intent
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r.correct:
            categories[cat]["correct"] += 1

    print("\n  Per-category accuracy:")
    for cat, stats in sorted(categories.items()):
        cat_acc = stats["correct"] / stats["total"] * 100
        print("    %-15s %d/%d (%.0f%%)" % (
            cat, stats["correct"], stats["total"], cat_acc))

    # Confusion: most common errors
    errors = [(r.utterance[:40], r.expected_intent, r.predicted_intent)
              for r in results if not r.correct]
    if errors:
        print("\n  Errors:")
        for utterance, expected, got in errors[:10]:
            print("    %-40s  expected=%-20s got=%s" % (utterance, expected, got))

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "parse_fails": parse_fails,
        "avg_latency_ms": avg_time,
    }


def compare_models(model_name: str = "Qwen/Qwen3-4B-MLX-4bit",
                    adapter_path: str = "~/.kat/adapters") -> None:
    """Run A/B comparison between base and fine-tuned model"""
    import os
    adapter_path = os.path.expanduser(adapter_path)

    print("=" * 60)
    print("  A/B Comparison: Base vs Fine-tuned")
    print("=" * 60)

    # Base model
    print("\n--- Base Model ---")
    base_results = evaluate_model(model_name=model_name, adapter_path=None)
    base_report = print_report(base_results, label="Base Model")

    # Fine-tuned model
    if os.path.exists(adapter_path):
        print("\n--- Fine-tuned Model ---")
        ft_results = evaluate_model(model_name=model_name, adapter_path=adapter_path)
        ft_report = print_report(ft_results, label="Fine-tuned Model")

        # Summary comparison
        print("\n" + "=" * 60)
        print("  Summary Comparison")
        print("=" * 60)
        print("  %-20s  %10s  %10s" % ("Metric", "Base", "Fine-tuned"))
        print("  " + "-" * 44)
        print("  %-20s  %9.1f%%  %9.1f%%" % (
            "Accuracy", base_report["accuracy"], ft_report["accuracy"]))
        print("  %-20s  %8.0fms  %8.0fms" % (
            "Avg Latency", base_report["avg_latency_ms"], ft_report["avg_latency_ms"]))
        print("  %-20s  %10d  %10d" % (
            "Parse Failures", base_report["parse_fails"], ft_report["parse_fails"]))

        delta = ft_report["accuracy"] - base_report["accuracy"]
        print("\n  Accuracy delta: %+.1f%%" % delta)
    else:
        print("\nNo adapter found at %s" % adapter_path)
        print("Run training first: python -m jarvis.training.train --train")


def main():
    """CLI for model evaluation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Kat voice routing accuracy")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--compare", action="store_true",
                        help="A/B comparison between base and fine-tuned")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-MLX-4bit",
                        help="Base model name")

    args = parser.parse_args()

    if args.compare:
        compare_models(
            model_name=args.model,
            adapter_path=args.adapter or "~/.kat/adapters",
        )
    else:
        results = evaluate_model(
            model_name=args.model,
            adapter_path=args.adapter,
        )
        print_report(results, label="Base" if not args.adapter else "Fine-tuned")


if __name__ == "__main__":
    main()
