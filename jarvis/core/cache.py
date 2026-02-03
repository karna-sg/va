"""
Simple in-memory cache for Jarvis

Caches frequently accessed data like GitHub issues, commits
to provide instant responses for repeated queries.
"""

import time
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """A single cache entry with expiration"""
    value: Any
    expires_at: float

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class SimpleCache:
    """
    Simple in-memory cache with TTL support.

    Used to cache GitHub data, reducing API calls and
    providing faster responses for repeated queries.
    """

    def __init__(self, default_ttl: float = 300.0):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds (5 minutes)
        """
        self._cache: dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        ttl = ttl if ttl is not None else self.default_ttl
        self._cache[key] = CacheEntry(
            value=value,
            expires_at=time.time() + ttl
        )

    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached data"""
        self._cache.clear()

    def cleanup(self) -> int:
        """Remove expired entries, return count of removed"""
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired:
            del self._cache[key]
        return len(expired)

    @property
    def stats(self) -> dict:
        """Get cache statistics"""
        return {
            'size': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        }


# Global cache instance
_cache: Optional[SimpleCache] = None


def get_cache() -> SimpleCache:
    """Get the global cache instance"""
    global _cache
    if _cache is None:
        _cache = SimpleCache(default_ttl=300.0)  # 5 minute default
    return _cache
