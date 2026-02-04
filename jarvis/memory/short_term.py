"""
Short-term Memory for Jarvis

In-memory conversation buffer with entity tracking.
Enables features like "send this to Slack" (references last result).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict, List


@dataclass
class MemoryEntry:
    """A single memory entry"""
    key: str
    value: Any
    category: str  # 'entity', 'tool_result', 'fact', 'preference'
    timestamp: datetime = field(default_factory=datetime.now)
    ttl_seconds: Optional[float] = None  # None = no expiry

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.ttl_seconds


class ShortTermMemory:
    """
    In-memory short-term storage.

    Stores:
    - Recent tool results (for reference resolution)
    - Session facts (current repo, channel, person)
    - Temporary preferences
    """

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._entries: Dict[str, MemoryEntry] = {}
        self._recent_results: List[Dict[str, Any]] = []

    def store(self, key: str, value: Any, category: str = 'fact',
              ttl: Optional[float] = None) -> None:
        """Store a memory entry"""
        self._entries[key] = MemoryEntry(
            key=key,
            value=value,
            category=category,
            ttl_seconds=ttl,
        )
        self._cleanup()

    def recall(self, key: str) -> Optional[Any]:
        """Recall a memory entry by key"""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._entries[key]
            return None
        return entry.value

    def recall_by_category(self, category: str) -> List[MemoryEntry]:
        """Get all entries of a category"""
        self._cleanup()
        return [e for e in self._entries.values() if e.category == category]

    def store_tool_result(self, tool_name: str, result: Any,
                          summary: str = "") -> None:
        """Store a tool result for reference"""
        self._recent_results.append({
            'tool': tool_name,
            'result': result,
            'summary': summary,
            'timestamp': datetime.now(),
        })
        if len(self._recent_results) > 20:
            self._recent_results = self._recent_results[-20:]

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent tool result"""
        if self._recent_results:
            return self._recent_results[-1]
        return None

    def get_context_for_llm(self) -> str:
        """Build context string for LLM injection"""
        parts = []

        # Active entities
        entities = self.recall_by_category('entity')
        if entities:
            entity_parts = ["%s: %s" % (e.key, e.value) for e in entities[-5:]]
            parts.append("Active context: %s" % ", ".join(entity_parts))

        # Last tool result
        last = self.get_last_result()
        if last:
            parts.append("Last tool: %s (%s)" % (last['tool'], last.get('summary', 'no summary')))

        # Preferences
        prefs = self.recall_by_category('preference')
        if prefs:
            pref_parts = ["%s: %s" % (p.key, p.value) for p in prefs[-3:]]
            parts.append("Preferences: %s" % ", ".join(pref_parts))

        return "; ".join(parts) if parts else ""

    def clear(self) -> None:
        """Clear all memory"""
        self._entries.clear()
        self._recent_results.clear()

    def _cleanup(self) -> None:
        """Remove expired entries and enforce max size"""
        # Remove expired
        expired = [k for k, v in self._entries.items() if v.is_expired]
        for key in expired:
            del self._entries[key]

        # Enforce max size (remove oldest)
        if len(self._entries) > self.max_entries:
            sorted_entries = sorted(
                self._entries.items(),
                key=lambda x: x[1].timestamp
            )
            to_remove = len(self._entries) - self.max_entries
            for key, _ in sorted_entries[:to_remove]:
                del self._entries[key]
