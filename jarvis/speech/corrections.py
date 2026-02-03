"""
Speech Transcription Corrections for Jarvis

Handles common speech-to-text errors and normalizes input
for better understanding.
"""

import re
from typing import Optional


# Common misheard words and their corrections
WORD_CORRECTIONS = {
    # Project names
    "curious": "curiescious",
    "curios": "curiescious",
    "curious cious": "curiescious",
    "curiouscious": "curiescious",

    # Technical terms
    "git hub": "github",
    "get hub": "github",
    "good hub": "github",
    "gith up": "github",
    "repo": "repository",
    "reppo": "repository",
    "pr": "pull request",
    "prs": "pull requests",
    "p r": "pull request",

    # Assistant name
    "jarrus": "jarvis",
    "jarvas": "jarvis",
    "jarves": "jarvis",
    "service": "jarvis",
    "hey jarvis": "jarvis",

    # Common commands
    "summit": "commit",
    "summits": "commits",
    "comit": "commit",
    "brunch": "branch",
    "marge": "merge",
    "poll request": "pull request",
    "poor request": "pull request",

    # Numbers (speech often misses)
    "issue eighteen": "issue 18",
    "issue seventeen": "issue 17",
    "issue sixteen": "issue 16",
    "issue fifteen": "issue 15",
    "issue fourteen": "issue 14",
    "issue thirteen": "issue 13",
    "issue twelve": "issue 12",
    "issue eleven": "issue 11",
    "issue ten": "issue 10",
    "issue nine": "issue 9",
    "issue eight": "issue 8",
    "issue seven": "issue 7",
    "issue six": "issue 6",
    "issue five": "issue 5",
    "issue four": "issue 4",
    "issue three": "issue 3",
    "issue two": "issue 2",
    "issue one": "issue 1",

    # Time expressions
    "yester day": "yesterday",
    "to day": "today",
    "to morrow": "tomorrow",
    "last week": "last week",

    # Actions
    "check out": "checkout",
    "check in": "checkin",
    "log in": "login",
    "log out": "logout",
    "set up": "setup",
    "start up": "startup",
}

# Patterns to normalize (regex-based)
PATTERN_CORRECTIONS = [
    # Normalize "issue number X" to "issue X"
    (r"issue\s+number\s+(\d+)", r"issue \1"),
    # Normalize "PR number X" to "PR X"
    (r"p\.?r\.?\s+number\s+(\d+)", r"pull request \1"),
    # Remove filler words
    (r"\b(um|uh|like|you know|basically)\b", ""),
    # Normalize multiple spaces
    (r"\s+", " "),
]


def correct_transcription(text: str) -> str:
    """
    Correct common speech-to-text errors.

    Args:
        text: Raw transcription from STT

    Returns:
        Corrected text
    """
    if not text:
        return text

    # Convert to lowercase for matching
    result = text.lower().strip()

    # Apply word corrections
    for wrong, correct in WORD_CORRECTIONS.items():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(wrong) + r'\b'
        result = re.sub(pattern, correct, result, flags=re.IGNORECASE)

    # Apply pattern corrections
    for pattern, replacement in PATTERN_CORRECTIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Clean up
    result = result.strip()
    result = re.sub(r'\s+', ' ', result)

    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]

    return result


def extract_intent(text: str) -> Optional[dict]:
    """
    Try to extract clear intent from potentially garbled input.

    Returns:
        Dict with 'action' and 'target' if intent is clear, None otherwise
    """
    text_lower = text.lower()

    # GitHub-related intents
    if any(w in text_lower for w in ['commit', 'summit', 'comit']):
        return {'action': 'check_commits', 'target': 'github'}

    if any(w in text_lower for w in ['pull request', 'pr', 'poll request']):
        return {'action': 'check_prs', 'target': 'github'}

    if any(w in text_lower for w in ['issue', 'issues']):
        # Try to extract issue number
        match = re.search(r'issue\s*#?\s*(\d+)', text_lower)
        if match:
            return {'action': 'get_issue', 'target': 'github', 'number': int(match.group(1))}
        return {'action': 'list_issues', 'target': 'github'}

    if any(w in text_lower for w in ['yesterday', 'today', 'what i did', 'what we did']):
        return {'action': 'check_activity', 'target': 'github'}

    return None


# Test
if __name__ == "__main__":
    test_cases = [
        "Hey Jarrus, what we did yester day?",
        "Check my git hub summits",
        "Show me issue eighteen on curious",
        "Um, like, can you check the prs?",
        "What's the status of poll request number five?",
    ]

    print("Testing transcription corrections:\n")
    for test in test_cases:
        corrected = correct_transcription(test)
        intent = extract_intent(corrected)
        print(f"Original:  {test}")
        print(f"Corrected: {corrected}")
        print(f"Intent:    {intent}")
        print()
