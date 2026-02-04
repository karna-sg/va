"""
Smart Speech Buffer for Jarvis

Handles text buffering and determines when to trigger speech.
Ensures reliable TTS even with incomplete or markdown-heavy responses.
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class SpeechBuffer:
    """
    Intelligent buffer that collects text and determines when to speak.

    Triggers speech when:
    1. Complete sentence detected (ends with . ! ?)
    2. Colon followed by enough content
    3. Enough words accumulated (fallback)
    4. Timeout reached (emergency fallback)

    OPTIMIZED for low latency - smaller chunks = faster first audio
    """

    min_words: int = 4  # Reduced from 8 - speak sooner
    max_words: int = 18  # Reduced from 30 - smaller chunks
    timeout_seconds: float = 1.5  # Reduced from 3.0 - faster fallback

    _buffer: str = ""
    _last_add_time: float = field(default_factory=time.time)
    _total_spoken: int = 0

    def add(self, text: str) -> None:
        """Add text to buffer"""
        self._buffer += text
        self._last_add_time = time.time()

    def get_speakable(self) -> Optional[str]:
        """
        Check if buffer has speakable content.
        Returns cleaned text to speak, or None if should wait.
        """
        if not self._buffer.strip():
            return None

        # Clean the buffer for speech
        cleaned = self._clean_for_speech(self._buffer)
        if not cleaned or len(cleaned) < 3:
            return None

        word_count = len(cleaned.split())

        # Check for complete sentence
        sentence_end = self._find_sentence_end(cleaned)
        if sentence_end > 0:
            to_speak = cleaned[:sentence_end + 1].strip()
            remaining = cleaned[sentence_end + 1:].strip()
            if to_speak:
                self._buffer = remaining
                self._total_spoken += 1
                return to_speak

        # Check for colon with content after it
        if ':' in cleaned:
            colon_pos = cleaned.rfind(':')
            after_colon = cleaned[colon_pos + 1:].strip()
            if len(after_colon.split()) >= 3:  # At least 3 words after colon
                to_speak = cleaned[:colon_pos + 1].strip()
                self._buffer = after_colon
                self._total_spoken += 1
                return to_speak

        # Force speak if we have enough words
        if word_count >= self.max_words:
            # Find a good break point
            words = cleaned.split()
            to_speak = ' '.join(words[:self.max_words])
            remaining = ' '.join(words[self.max_words:])
            self._buffer = remaining
            self._total_spoken += 1
            return to_speak

        # Force speak on timeout (if we have minimum words)
        if word_count >= self.min_words:
            elapsed = time.time() - self._last_add_time
            if elapsed >= self.timeout_seconds:
                self._buffer = ""
                self._total_spoken += 1
                return cleaned

        return None

    def flush(self) -> Optional[str]:
        """Force get remaining content (call at end of stream)"""
        if not self._buffer.strip():
            return None
        cleaned = self._clean_for_speech(self._buffer)
        self._buffer = ""
        if cleaned and len(cleaned) > 3:
            self._total_spoken += 1
            return cleaned
        return None

    def has_spoken(self) -> bool:
        """Check if any content has been spoken"""
        return self._total_spoken > 0

    def reset(self) -> None:
        """Reset buffer state"""
        self._buffer = ""
        self._total_spoken = 0
        self._last_add_time = time.time()

    def _find_sentence_end(self, text: str) -> int:
        """Find position of sentence-ending punctuation"""
        # Look for . ! ? followed by space or end
        for i, char in enumerate(text):
            if char in '.!?':
                # Check if it's really end of sentence
                # (not abbreviation like "Dr." or "e.g.")
                if i == len(text) - 1:
                    return i
                next_char = text[i + 1] if i + 1 < len(text) else ' '
                if next_char in ' \n\t':
                    # Check it's not an abbreviation
                    before = text[max(0, i-3):i].lower()
                    if before not in ['dr.', 'mr.', 'ms.', 'vs.', 'e.g', 'i.e']:
                        return i
        return -1

    def _clean_for_speech(self, text: str) -> str:
        """Clean text for TTS output"""
        # Remove URLs
        text = re.sub(r'https?://[^\s]+', '', text)

        # Remove file paths
        text = re.sub(r'[/\\][\w/\\.-]+\.\w+', '', text)
        text = re.sub(r'[/\\]Users[/\\][^\s]+', '', text)

        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)

        # Remove markdown formatting but keep text
        text = text.replace('**', '')
        text = text.replace('*', '')
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

        # Keep link text, remove URL
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Remove tables
        text = re.sub(r'\|[^\n]+\|', '', text)

        # Clean list markers but keep content
        text = re.sub(r'^[\s]*[-*•]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s*', '', text, flags=re.MULTILINE)

        # Remove emoji
        text = re.sub(r'[🔴🟡🟢✅❌📝🎤💬🚀📊👥🗺️🧪👋🏗️📚📖🔬🔐🔍📤✏️📱🎨🔗🎙️]', '', text)

        # Remove commit hashes
        text = re.sub(r'#\d+', lambda m: f"number {m.group()[1:]}", text)  # #123 -> "number 123"
        text = re.sub(r'\b[a-f0-9]{7,40}\b', '', text)

        # Replace symbols
        text = text.replace('&', ' and ')
        text = text.replace('@', ' at ')
        text = text.replace('%', ' percent ')
        text = text.replace('→', ' to ')
        text = text.replace('--', ', ')
        text = text.replace('_', ' ')

        # Normalize whitespace
        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r'\s+', ' ', text)

        # Clean up punctuation
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r',+', ',', text)
        text = re.sub(r'\.\s*\.', '.', text)
        text = re.sub(r':\s*\.', ':', text)
        text = re.sub(r'\s+([.,!?:])', r'\1', text)

        return text.strip()


def get_acknowledgment(query: str, user_name: str = "Vasu") -> str:
    """
    Get context-aware acknowledgment based on query type.
    Returns a SHORT phrase to speak immediately.
    Uses user_name naturally (not every time).
    """
    query_lower = query.lower()

    # GitHub-related
    if any(w in query_lower for w in ['github', 'commit', 'summit', 'issue', 'pr', 'pull request', 'repo']):
        return "Checking GitHub, %s." % user_name

    # Time-based queries
    if any(w in query_lower for w in ['yesterday', 'today', 'recent', 'latest', 'last week']):
        return "Let me check."

    # Implementation requests
    if any(w in query_lower for w in ['implement', 'create', 'build', 'make', 'add', 'write']):
        return "On it, %s." % user_name

    # Fix/debug requests
    if any(w in query_lower for w in ['fix', 'debug', 'solve', 'resolve']):
        return "On it."

    # Questions
    if any(w in query_lower for w in ['what', 'how', 'why', 'where', 'when', 'which', 'explain']):
        return "Let me see."

    # List/show requests
    if any(w in query_lower for w in ['list', 'show', 'display', 'get']):
        return "Getting that."

    # Default
    return "Sure, %s." % user_name


# Test
if __name__ == "__main__":
    # Test acknowledgments
    test_queries = [
        "what did we do yesterday",
        "show me open issues",
        "implement issue 5",
        "fix the bug in login",
        "hello jarvis",
    ]

    print("Testing acknowledgments:")
    for q in test_queries:
        ack = get_acknowledgment(q)
        print(f"  '{q}' -> '{ack}'")

    # Test buffer
    print("\nTesting speech buffer:")
    buffer = SpeechBuffer()

    test_text = "Here are the issues. First is about setup. Second is about login. Third involves the dashboard."

    for word in test_text.split():
        buffer.add(word + " ")
        speakable = buffer.get_speakable()
        if speakable:
            print(f"  SPEAK: '{speakable}'")

    remaining = buffer.flush()
    if remaining:
        print(f"  FLUSH: '{remaining}'")
