"""
Temporal Memory for Jarvis

Time-bound facts with automatic expiry and smart recency.

Handles facts like:
- "I'm working on the auth feature" (expires in hours)
- "We're in sprint 5" (expires in weeks)
- "The deploy is at 3pm" (expires at specific time)
- "John is on vacation until Friday" (expires on date)

Also tracks time-of-day patterns:
- Morning routine commands
- End-of-day summaries
- Weekly review habits
"""

import time
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field


@dataclass
class TemporalFact:
    """A time-bound fact"""
    id: int
    key: str
    content: str
    fact_type: str  # 'session', 'daily', 'weekly', 'custom', 'permanent'
    metadata: Dict[str, Any]
    created_at: float
    expires_at: Optional[float]  # Unix timestamp, None = no expiry
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def time_remaining(self) -> Optional[float]:
        """Seconds until expiry, or None if no expiry"""
        if self.expires_at is None:
            return None
        return max(0, self.expires_at - time.time())

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


# Default TTL values for fact types
FACT_TTL = {
    'session': 4 * 3600,       # 4 hours
    'daily': 24 * 3600,        # 24 hours
    'weekly': 7 * 24 * 3600,   # 7 days
    'monthly': 30 * 24 * 3600, # 30 days
    'permanent': None,          # No expiry
}


class TemporalMemory:
    """
    Time-aware fact storage with automatic expiry.

    Facts have different lifetimes based on their type:
    - session: Current work session (~4 hours)
    - daily: Today's context (24 hours)
    - weekly: This sprint/week (7 days)
    - custom: User-defined TTL
    - permanent: Never expires

    Also tracks temporal patterns (time-of-day, day-of-week).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else (
            Path.home() / '.jarvis' / 'memory.db'
        )
        self._conn = None  # type: Optional[sqlite3.Connection]

    def initialize(self) -> bool:
        """Initialize database tables"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

            self._conn.executescript('''
                CREATE TABLE IF NOT EXISTS temporal_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    fact_type TEXT NOT NULL DEFAULT 'session',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    access_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS temporal_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    hour_of_day INTEGER,
                    day_of_week INTEGER,
                    command TEXT NOT NULL,
                    intent TEXT NOT NULL DEFAULT '',
                    count INTEGER NOT NULL DEFAULT 1,
                    last_seen REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tf_key ON temporal_facts(key);
                CREATE INDEX IF NOT EXISTS idx_tf_type ON temporal_facts(fact_type);
                CREATE INDEX IF NOT EXISTS idx_tf_expires ON temporal_facts(expires_at);
                CREATE INDEX IF NOT EXISTS idx_tp_hour ON temporal_patterns(hour_of_day);
                CREATE INDEX IF NOT EXISTS idx_tp_day ON temporal_patterns(day_of_week);
            ''')
            self._conn.commit()
            return True

        except Exception as e:
            print("  Temporal memory error: %s" % e)
            return False

    # -------------------------------------------------------------------------
    # Fact Storage
    # -------------------------------------------------------------------------

    def store_fact(self, key: str, content: str,
                   fact_type: str = 'session',
                   metadata: Optional[Dict[str, Any]] = None,
                   ttl_seconds: Optional[float] = None) -> Optional[int]:
        """
        Store a temporal fact.

        Args:
            key: Unique identifier (e.g. 'current_task', 'sprint_number')
            content: The fact content
            fact_type: 'session', 'daily', 'weekly', 'monthly', 'permanent', 'custom'
            metadata: Additional structured data
            ttl_seconds: Override TTL (for 'custom' type)

        Returns:
            Row ID or None
        """
        if not self._conn:
            return None

        now = time.time()

        # Calculate expiry
        if ttl_seconds is not None:
            expires_at = now + ttl_seconds
        elif fact_type in FACT_TTL:
            ttl = FACT_TTL[fact_type]
            expires_at = (now + ttl) if ttl else None
        else:
            expires_at = now + FACT_TTL['session']

        meta_json = json.dumps(metadata or {})

        # Upsert
        self._conn.execute('''
            INSERT INTO temporal_facts
            (key, content, fact_type, metadata_json, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                content = excluded.content,
                fact_type = excluded.fact_type,
                metadata_json = excluded.metadata_json,
                expires_at = excluded.expires_at,
                access_count = access_count + 1
        ''', (key, content, fact_type, meta_json, now, expires_at))
        self._conn.commit()

        return self._conn.execute(
            'SELECT id FROM temporal_facts WHERE key = ?', (key,)
        ).fetchone()['id']

    def get_fact(self, key: str) -> Optional[TemporalFact]:
        """Get a fact by key (returns None if expired)"""
        if not self._conn:
            return None

        row = self._conn.execute(
            'SELECT * FROM temporal_facts WHERE key = ?', (key,)
        ).fetchone()

        if not row:
            return None

        fact = self._row_to_fact(row)

        if fact.is_expired:
            self._conn.execute('DELETE FROM temporal_facts WHERE key = ?', (key,))
            self._conn.commit()
            return None

        # Update access count
        self._conn.execute(
            'UPDATE temporal_facts SET access_count = access_count + 1 WHERE key = ?',
            (key,)
        )
        self._conn.commit()

        return fact

    def get_active_facts(self, fact_type: Optional[str] = None,
                         limit: int = 20) -> List[TemporalFact]:
        """Get all non-expired facts, optionally filtered by type"""
        if not self._conn:
            return []

        now = time.time()

        if fact_type:
            rows = self._conn.execute('''
                SELECT * FROM temporal_facts
                WHERE fact_type = ? AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY created_at DESC LIMIT ?
            ''', (fact_type, now, limit)).fetchall()
        else:
            rows = self._conn.execute('''
                SELECT * FROM temporal_facts
                WHERE expires_at IS NULL OR expires_at > ?
                ORDER BY created_at DESC LIMIT ?
            ''', (now, limit)).fetchall()

        return [self._row_to_fact(row) for row in rows]

    def remove_fact(self, key: str) -> bool:
        """Remove a fact by key"""
        if not self._conn:
            return False

        cursor = self._conn.execute(
            'DELETE FROM temporal_facts WHERE key = ?', (key,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def extend_fact(self, key: str, additional_seconds: float) -> bool:
        """Extend the TTL of an existing fact"""
        if not self._conn:
            return False

        row = self._conn.execute(
            'SELECT expires_at FROM temporal_facts WHERE key = ?', (key,)
        ).fetchone()

        if not row or row['expires_at'] is None:
            return False

        new_expires = max(row['expires_at'], time.time()) + additional_seconds
        self._conn.execute(
            'UPDATE temporal_facts SET expires_at = ? WHERE key = ?',
            (new_expires, key)
        )
        self._conn.commit()
        return True

    # -------------------------------------------------------------------------
    # Temporal Patterns
    # -------------------------------------------------------------------------

    def record_command(self, command: str, intent: str = '') -> None:
        """
        Record a command for temporal pattern analysis.

        Tracks what commands are used at what time of day / day of week.
        """
        if not self._conn:
            return

        now = datetime.now()
        hour = now.hour
        day = now.weekday()  # 0=Monday, 6=Sunday

        # Upsert pattern
        existing = self._conn.execute('''
            SELECT id, count FROM temporal_patterns
            WHERE hour_of_day = ? AND day_of_week = ? AND command = ?
        ''', (hour, day, command)).fetchone()

        if existing:
            self._conn.execute('''
                UPDATE temporal_patterns
                SET count = count + 1, last_seen = ?, intent = ?
                WHERE id = ?
            ''', (time.time(), intent, existing['id']))
        else:
            self._conn.execute('''
                INSERT INTO temporal_patterns
                (pattern_type, hour_of_day, day_of_week, command, intent, count, last_seen)
                VALUES ('hourly', ?, ?, ?, ?, 1, ?)
            ''', (hour, day, command, intent, time.time()))

        self._conn.commit()

    def get_likely_commands(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get commands the user likely wants based on current time.

        Returns commands commonly used at this hour/day.
        """
        if not self._conn:
            return []

        now = datetime.now()
        hour = now.hour
        day = now.weekday()

        # Commands at this hour on this day
        rows = self._conn.execute('''
            SELECT command, intent, count, last_seen FROM temporal_patterns
            WHERE hour_of_day = ? AND day_of_week = ?
            ORDER BY count DESC
            LIMIT ?
        ''', (hour, day, limit)).fetchall()

        if not rows:
            # Fall back to commands at this hour on any day
            rows = self._conn.execute('''
                SELECT command, intent, SUM(count) as count, MAX(last_seen) as last_seen
                FROM temporal_patterns
                WHERE hour_of_day = ?
                GROUP BY command, intent
                ORDER BY count DESC
                LIMIT ?
            ''', (hour, limit)).fetchall()

        return [
            {
                'command': row['command'],
                'intent': row['intent'],
                'count': row['count'],
                'last_seen': row['last_seen'],
            }
            for row in rows
        ]

    def get_daily_pattern(self) -> List[Dict[str, Any]]:
        """Get the user's typical daily command pattern"""
        if not self._conn:
            return []

        rows = self._conn.execute('''
            SELECT hour_of_day, command, intent, SUM(count) as total_count
            FROM temporal_patterns
            GROUP BY hour_of_day, command
            HAVING total_count >= 3
            ORDER BY hour_of_day, total_count DESC
        ''').fetchall()

        return [
            {
                'hour': row['hour_of_day'],
                'command': row['command'],
                'intent': row['intent'],
                'count': row['total_count'],
            }
            for row in rows
        ]

    # -------------------------------------------------------------------------
    # Context for LLM
    # -------------------------------------------------------------------------

    def get_context_for_llm(self) -> str:
        """Build context string from active temporal facts"""
        parts = []

        # Active session facts
        session_facts = self.get_active_facts('session', limit=5)
        if session_facts:
            fact_strs = ["%s: %s" % (f.key, f.content) for f in session_facts]
            parts.append("Current session: %s" % "; ".join(fact_strs))

        # Active daily facts
        daily_facts = self.get_active_facts('daily', limit=3)
        if daily_facts:
            fact_strs = ["%s: %s" % (f.key, f.content) for f in daily_facts]
            parts.append("Today: %s" % "; ".join(fact_strs))

        # Time context
        now = datetime.now()
        likely = self.get_likely_commands(limit=2)
        if likely:
            cmd_strs = [c['command'] for c in likely]
            parts.append("Usually at %d:00: %s" % (now.hour, ", ".join(cmd_strs)))

        return "\n".join(parts) if parts else ""

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove all expired facts"""
        if not self._conn:
            return 0

        cursor = self._conn.execute(
            'DELETE FROM temporal_facts WHERE expires_at IS NOT NULL AND expires_at < ?',
            (time.time(),)
        )
        self._conn.commit()
        return cursor.rowcount

    def cleanup_old_patterns(self, max_age_days: int = 90) -> int:
        """Remove patterns not seen in max_age_days"""
        if not self._conn:
            return 0

        cutoff = time.time() - (max_age_days * 86400)
        cursor = self._conn.execute(
            'DELETE FROM temporal_patterns WHERE last_seen < ? AND count < 5',
            (cutoff,)
        )
        self._conn.commit()
        return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """Get temporal memory statistics"""
        if not self._conn:
            return {'status': 'not initialized'}

        now = time.time()

        fact_row = self._conn.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN expires_at IS NULL OR expires_at > ? THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN expires_at IS NOT NULL AND expires_at <= ? THEN 1 ELSE 0 END) as expired
            FROM temporal_facts
        ''', (now, now)).fetchone()

        pattern_row = self._conn.execute('''
            SELECT COUNT(*) as total, SUM(count) as total_commands
            FROM temporal_patterns
        ''').fetchone()

        return {
            'facts_total': fact_row['total'] or 0,
            'facts_active': fact_row['active'] or 0,
            'facts_expired': fact_row['expired'] or 0,
            'patterns_total': pattern_row['total'] or 0,
            'commands_tracked': pattern_row['total_commands'] or 0,
        }

    def _row_to_fact(self, row) -> TemporalFact:
        """Convert SQLite row to TemporalFact"""
        return TemporalFact(
            id=row['id'],
            key=row['key'],
            content=row['content'],
            fact_type=row['fact_type'],
            metadata=json.loads(row['metadata_json']),
            created_at=row['created_at'],
            expires_at=row['expires_at'],
            access_count=row['access_count'],
        )

    def shutdown(self) -> None:
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
