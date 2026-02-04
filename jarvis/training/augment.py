"""
Training Data Augmentation for Kat Voice Assistant

Generates paraphrases and variations of seed training samples to expand
the fine-tuning dataset. Uses rule-based augmentation (no LLM needed):
1. Synonym substitution
2. Word reordering
3. Filler word injection (voice-realistic)
4. Pronoun/name variations
5. Prefix/suffix variations

Typically expands a ~150 seed dataset to ~500+ training samples.
"""

import json
import random
from typing import List, Dict, Any, Optional


# Voice-realistic filler words and prefixes
VOICE_PREFIXES = [
    "", "hey kat ", "kat ", "hey ", "okay ",
    "can you ", "could you ", "please ",
    "I need to ", "I want to ", "let's ",
]

# Synonym groups for substitution
SYNONYMS = {
    "show": ["show me", "display", "list", "get", "pull up", "check"],
    "issues": ["issues", "tickets", "bugs", "tasks"],
    "pull requests": ["pull requests", "PRs", "merge requests"],
    "commits": ["commits", "changes", "pushes"],
    "status": ["status", "state", "progress", "update"],
    "send": ["send", "post", "share", "push"],
    "run": ["run", "execute", "start", "kick off"],
    "fix": ["fix", "resolve", "debug", "patch"],
    "implement": ["implement", "build", "create", "develop", "code"],
    "review": ["review", "check", "look at", "analyze", "examine"],
    "explain": ["explain", "describe", "walk me through", "break down"],
    "refactor": ["refactor", "clean up", "simplify", "improve", "restructure"],
}

# Intent-specific context variations
REPO_REFS = [
    "curiescious", "the repo", "our repo", "the project",
    "our project", "curiescious repo",
]

CHANNEL_REFS = [
    "general", "engineering", "the team channel",
    "the main channel", "slack",
]


def augment_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate augmented variations of a single training sample"""
    augmented = []
    utterance = sample["utterance"]
    intent = sample["intent"]

    # 1. Prefix variations (add voice-realistic prefixes)
    for prefix in random.sample(VOICE_PREFIXES, min(3, len(VOICE_PREFIXES))):
        if prefix and not utterance.lower().startswith(prefix.strip()):
            new_utterance = (prefix + utterance).strip()
            augmented.append(_make_sample(new_utterance, sample))

    # 2. Synonym substitution
    lower = utterance.lower()
    for word, syns in SYNONYMS.items():
        if word in lower:
            for syn in random.sample(syns, min(2, len(syns))):
                if syn != word:
                    new_utterance = lower.replace(word, syn, 1)
                    augmented.append(_make_sample(new_utterance, sample))

    # 3. Repo reference variations (for GitHub intents)
    if intent.startswith("github.") or intent.startswith("workflow."):
        for repo_ref in random.sample(REPO_REFS, min(2, len(REPO_REFS))):
            if "curiescious" in lower and repo_ref != "curiescious":
                new_utterance = lower.replace("curiescious", repo_ref)
                augmented.append(_make_sample(new_utterance, sample))
            elif "curiescious" not in lower and "repo" not in lower:
                new_utterance = utterance + " on " + repo_ref
                augmented.append(_make_sample(new_utterance, sample))

    # 4. Channel variations (for Slack intents)
    if intent.startswith("slack.") or intent == "workflow.daily_status":
        for channel in random.sample(CHANNEL_REFS, min(2, len(CHANNEL_REFS))):
            if "general" in lower and channel != "general":
                new_utterance = lower.replace("general", channel)
                new_params = dict(sample["params"])
                if "channel" in new_params:
                    new_params["channel"] = channel.replace("the ", "").replace(" channel", "")
                augmented.append(_make_sample(new_utterance, sample, params=new_params))

    # 5. Casual voice variations
    casual_suffixes = [
        " please", " for me", " real quick", " when you get a chance",
    ]
    suffix = random.choice(casual_suffixes)
    augmented.append(_make_sample(utterance + suffix, sample))

    # Deduplicate by utterance text
    seen = {sample["utterance"].lower()}
    unique = []
    for aug in augmented:
        key = aug["utterance"].lower().strip()
        if key not in seen and len(key) > 3:
            seen.add(key)
            unique.append(aug)

    return unique


def _make_sample(utterance: str, base: Dict[str, Any],
                 params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new sample based on an existing one with a new utterance"""
    return {
        "utterance": utterance.strip(),
        "intent": base["intent"],
        "params": params if params is not None else base["params"],
        "needs_claude": base["needs_claude"],
        "missing_slots": base["missing_slots"],
    }


def augment_dataset(samples: List[Dict[str, Any]],
                    target_multiplier: float = 3.0,
                    seed: int = 42) -> List[Dict[str, Any]]:
    """
    Augment an entire dataset.

    Args:
        samples: Original training samples
        target_multiplier: Target dataset size as multiple of original
        seed: Random seed for reproducibility

    Returns:
        Combined original + augmented samples
    """
    random.seed(seed)

    all_samples = list(samples)  # Keep originals
    target_total = int(len(samples) * target_multiplier)

    for sample in samples:
        augmented = augment_sample(sample)
        all_samples.extend(augmented)

    # Trim if we exceeded target
    if len(all_samples) > target_total:
        # Keep all originals, randomly sample from augmented
        originals = samples
        augmented_only = all_samples[len(samples):]
        random.shuffle(augmented_only)
        keep_count = target_total - len(originals)
        all_samples = originals + augmented_only[:keep_count]

    return all_samples


def export_augmented_jsonl(samples: List[Dict[str, Any]],
                           output_path: str = "training_augmented.jsonl") -> int:
    """Export augmented samples as JSONL for LoRA fine-tuning"""
    with open(output_path, 'w') as f:
        for sample in samples:
            record = {
                "instruction": "Classify this voice command for Kat (Vasu's assistant). Output JSON.",
                "input": sample["utterance"],
                "output": json.dumps({
                    "intent": sample["intent"],
                    "confidence": 0.95,
                    "params": sample["params"],
                    "needs_claude": sample["needs_claude"],
                    "missing_slots": sample["missing_slots"],
                }),
            }
            f.write(json.dumps(record) + "\n")

    return len(samples)


if __name__ == "__main__":
    from jarvis.training.data import get_seed_samples

    seed_samples = get_seed_samples()
    print("Seed samples: %d" % len(seed_samples))

    augmented = augment_dataset(seed_samples, target_multiplier=3.5)
    print("After augmentation: %d" % len(augmented))

    count = export_augmented_jsonl(augmented)
    print("Exported %d samples to training_augmented.jsonl" % count)

    # Show intent distribution
    from collections import Counter
    dist = Counter(s["intent"] for s in augmented)
    print("\nIntent distribution:")
    for intent, cnt in dist.most_common(15):
        print("  %-30s %d" % (intent, cnt))
