"""
Session Store for Jarvis

SQLite-backed persistent storage for:
- Session facts
- Frequently accessed repos/channels/people
- Recent workflow outputs
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, Any, Dict, List
from dataclasses import dataclass


@dataclass
class SessionFact:
    """A persisted session fact"""
    key: str
    value: str
    category: str
    created_at: float
    updated_at: float
    access_count: int = 0


class SessionStore:
    """
    SQLite-backed session persistence.

    Stores at ~/.jarvis/memory.db:
    - Session facts (key-value with category)
    - Access frequency tracking
    - Auto-cleanup of stale entries
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path.home() / '.jarvis' / 'memory.db'

        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self) -> bool:
        """Initialize the database and create tables"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

            self._conn.executescript('''
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS workflow_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_name TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
                CREATE INDEX IF NOT EXISTS idx_facts_updated ON facts(updated_at);
                CREATE INDEX IF NOT EXISTS idx_workflow_created ON workflow_results(created_at);
            ''')
            self._conn.commit()

            print("  Session store: %s" % self.db_path)
            return True

        except Exception as e:
            print("  Session store error: %s" % e)
            return False

    def store_fact(self, key: str, value: Any, category: str = 'general') -> None:
        """Store or update a fact"""
        if not self._conn:
            return

        now = time.time()
        value_str = json.dumps(value) if not isinstance(value, str) else value

        self._conn.execute('''
            INSERT INTO facts (key, value, category, created_at, updated_at, access_count)
            VALUES (?, ?, ?, ?, ?, 0)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                category = excluded.category,
                updated_at = excluded.updated_at,
                access_count = access_count + 1
        ''', (key, value_str, category, now, now))
        self._conn.commit()

    def get_fact(self, key: str) -> Optional[str]:
        """Get a fact by key"""
        if not self._conn:
            return None

        row = self._conn.execute(
            'SELECT value FROM facts WHERE key = ?', (key,)
        ).fetchone()

        if row:
            self._conn.execute(
                'UPDATE facts SET access_count = access_count + 1, updated_at = ? WHERE key = ?',
                (time.time(), key)
            )
            self._conn.commit()
            return row['value']
        return None

    def get_facts_by_category(self, category: str, limit: int = 20) -> List[SessionFact]:
        """Get facts by category, ordered by most recently used"""
        if not self._conn:
            return []

        rows = self._conn.execute('''
            SELECT key, value, category, created_at, updated_at, access_count
            FROM facts WHERE category = ?
            ORDER BY updated_at DESC LIMIT ?
        ''', (category, limit)).fetchall()

        return [
            SessionFact(
                key=row['key'],
                value=row['value'],
                category=row['category'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                access_count=row['access_count'],
            )
            for row in rows
        ]

    def get_frequent(self, category: Optional[str] = None, limit: int = 10) -> List[SessionFact]:
        """Get most frequently accessed facts"""
        if not self._conn:
            return []

        if category:
            rows = self._conn.execute('''
                SELECT * FROM facts WHERE category = ?
                ORDER BY access_count DESC LIMIT ?
            ''', (category, limit)).fetchall()
        else:
            rows = self._conn.execute('''
                SELECT * FROM facts
                ORDER BY access_count DESC LIMIT ?
            ''', (limit,)).fetchall()

        return [
            SessionFact(
                key=row['key'],
                value=row['value'],
                category=row['category'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                access_count=row['access_count'],
            )
            for row in rows
        ]

    def store_workflow_result(self, workflow_name: str, result: Any) -> None:
        """Store a workflow execution result"""
        if not self._conn:
            return

        self._conn.execute('''
            INSERT INTO workflow_results (workflow_name, result_json, created_at)
            VALUES (?, ?, ?)
        ''', (workflow_name, json.dumps(result), time.time()))
        self._conn.commit()

    def get_recent_workflow_results(self, workflow_name: Optional[str] = None,
                                   limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent workflow results"""
        if not self._conn:
            return []

        if workflow_name:
            rows = self._conn.execute('''
                SELECT * FROM workflow_results WHERE workflow_name = ?
                ORDER BY created_at DESC LIMIT ?
            ''', (workflow_name, limit)).fetchall()
        else:
            rows = self._conn.execute('''
                SELECT * FROM workflow_results
                ORDER BY created_at DESC LIMIT ?
            ''', (limit,)).fetchall()

        return [
            {
                'workflow': row['workflow_name'],
                'result': json.loads(row['result_json']),
                'created_at': row['created_at'],
            }
            for row in rows
        ]

    def cleanup(self, max_age_days: int = 30) -> int:
        """Remove facts older than max_age_days with low access count"""
        if not self._conn:
            return 0

        cutoff = time.time() - (max_age_days * 86400)
        cursor = self._conn.execute('''
            DELETE FROM facts
            WHERE updated_at < ? AND access_count < 3
        ''', (cutoff,))

        self._conn.execute('''
            DELETE FROM workflow_results WHERE created_at < ?
        ''', (cutoff,))

        self._conn.commit()
        return cursor.rowcount

    async def shutdown(self) -> None:
        """Close the database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
