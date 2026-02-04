"""
Long-term Memory for Jarvis

FAISS-indexed persistent memory for:
- User preferences (coding style, default branches, preferred tools)
- Behavior patterns (common command sequences, time-of-day habits)
- Learned corrections (utterance -> intent mappings the user confirmed)
- Knowledge snippets (project-specific context extracted from conversations)

Uses the same sentence-transformers + FAISS stack as the intent router,
but for semantic recall of personal context.
"""

import json
import pickle
import time
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field


@dataclass
class MemoryRecord:
    """A long-term memory record"""
    id: int
    key: str
    content: str
    category: str  # 'preference', 'pattern', 'correction', 'knowledge'
    metadata: Dict[str, Any]
    created_at: float
    updated_at: float
    access_count: int = 0
    relevance_score: float = 0.0  # Set during retrieval


class LongTermMemory:
    """
    FAISS-indexed long-term memory with SQLite backing.

    Storage: SQLite at ~/.jarvis/memory.db (shared with SessionStore)
    Index: FAISS at ~/.jarvis/index/memory.faiss

    Operations:
    - store(): Add a memory with embedding for semantic search
    - recall(): Semantic search for relevant memories
    - recall_by_key(): Exact key lookup
    - get_preferences(): Get all user preferences
    - learn_correction(): Store a routing correction for future training
    """

    def __init__(self, db_path: Optional[str] = None,
                 index_dir: Optional[str] = None,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path) if db_path else (
            Path.home() / '.jarvis' / 'memory.db'
        )
        self.index_dir = Path(index_dir) if index_dir else (
            Path.home() / '.jarvis' / 'index'
        )
        self.model_name = model_name

        self._conn = None  # type: Optional[sqlite3.Connection]
        self._model = None
        self._index = None
        self._id_map = []  # type: List[int]  # Maps FAISS position -> SQLite row ID
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def initialize(self) -> bool:
        """Initialize database tables and load FAISS index"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

            self._conn.executescript('''
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'knowledge',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding BLOB,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    expires_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_ltm_key ON long_term_memory(key);
                CREATE INDEX IF NOT EXISTS idx_ltm_category ON long_term_memory(category);
                CREATE INDEX IF NOT EXISTS idx_ltm_updated ON long_term_memory(updated_at);
                CREATE INDEX IF NOT EXISTS idx_ltm_expires ON long_term_memory(expires_at);
            ''')
            self._conn.commit()

            # Try to load the embedding model and FAISS index
            self._load_model()
            self._load_index()

            self._is_ready = True
            return True

        except Exception as e:
            print("  Long-term memory error: %s" % e)
            return False

    def _load_model(self) -> bool:
        """Load the sentence-transformers model (shared with intent router)"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            return True
        except ImportError:
            # Graceful degradation - semantic search unavailable
            return False
        except Exception:
            return False

    def _load_index(self) -> bool:
        """Load existing FAISS index from disk"""
        faiss_path = self.index_dir / 'memory.faiss'
        map_path = self.index_dir / 'memory_id_map.pkl'

        if not faiss_path.exists() or not map_path.exists():
            return False

        try:
            import faiss

            self._index = faiss.read_index(str(faiss_path))

            with open(map_path, 'rb') as f:
                self._id_map = pickle.load(f)

            return True
        except Exception:
            return False

    def _save_index(self) -> bool:
        """Save FAISS index to disk"""
        if not self._index:
            return False

        try:
            import faiss

            self.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_dir / 'memory.faiss'))

            with open(self.index_dir / 'memory_id_map.pkl', 'wb') as f:
                pickle.dump(self._id_map, f)

            return True
        except Exception:
            return False

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Embed text using sentence-transformers"""
        if not self._model:
            return None

        embedding = self._model.encode([text], normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def _rebuild_index(self) -> bool:
        """Rebuild the FAISS index from all stored memories"""
        if not self._model or not self._conn:
            return False

        try:
            import faiss

            rows = self._conn.execute('''
                SELECT id, content FROM long_term_memory
                WHERE embedding IS NOT NULL
                ORDER BY id
            ''').fetchall()

            if not rows:
                return False

            # Re-embed all content
            contents = [row['content'] for row in rows]
            embeddings = self._model.encode(contents, normalize_embeddings=True,
                                            show_progress_bar=False)
            embeddings = np.array(embeddings, dtype=np.float32)

            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(embeddings)
            self._id_map = [row['id'] for row in rows]

            self._save_index()
            return True

        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def store(self, key: str, content: str, category: str = 'knowledge',
              metadata: Optional[Dict[str, Any]] = None,
              expires_in_days: Optional[float] = None) -> Optional[int]:
        """
        Store a memory record with optional FAISS embedding.

        Args:
            key: Identifier (e.g. 'pref.default_branch', 'pattern.morning_routine')
            content: The memory content (searchable text)
            category: 'preference', 'pattern', 'correction', 'knowledge'
            metadata: Additional structured data
            expires_in_days: Auto-expire after N days (None = never)

        Returns:
            Row ID of the stored record, or None on failure
        """
        if not self._conn:
            return None

        now = time.time()
        expires_at = (now + expires_in_days * 86400) if expires_in_days else None
        meta_json = json.dumps(metadata or {})

        # Embed the content
        embedding = self._embed(content)
        embedding_blob = embedding.tobytes() if embedding is not None else None

        # Check if key already exists (upsert)
        existing = self._conn.execute(
            'SELECT id FROM long_term_memory WHERE key = ?', (key,)
        ).fetchone()

        if existing:
            self._conn.execute('''
                UPDATE long_term_memory
                SET content = ?, category = ?, metadata_json = ?,
                    embedding = ?, updated_at = ?, expires_at = ?,
                    access_count = access_count + 1
                WHERE key = ?
            ''', (content, category, meta_json, embedding_blob, now, expires_at, key))
            self._conn.commit()
            row_id = existing['id']
        else:
            cursor = self._conn.execute('''
                INSERT INTO long_term_memory
                (key, content, category, metadata_json, embedding, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (key, content, category, meta_json, embedding_blob, now, now, expires_at))
            self._conn.commit()
            row_id = cursor.lastrowid

        # Update FAISS index
        if embedding is not None:
            self._add_to_index(row_id, embedding)

        return row_id

    def _add_to_index(self, row_id: int, embedding: np.ndarray) -> None:
        """Add or update a single vector in the FAISS index"""
        try:
            import faiss

            if self._index is None:
                dimension = embedding.shape[1]
                self._index = faiss.IndexFlatIP(dimension)
                self._id_map = []

            # For simplicity, rebuild periodically rather than updating in-place
            # FAISS IndexFlatIP doesn't support removal, so we just append
            if row_id in self._id_map:
                # Needs rebuild to update existing vector
                self._rebuild_index()
            else:
                self._index.add(embedding)
                self._id_map.append(row_id)
                self._save_index()

        except ImportError:
            pass

    def recall(self, query: str, top_k: int = 5,
               category: Optional[str] = None,
               min_score: float = 0.3) -> List[MemoryRecord]:
        """
        Semantic search for relevant memories.

        Args:
            query: Natural language query
            top_k: Max results
            category: Filter by category (None = all)
            min_score: Minimum cosine similarity threshold

        Returns:
            List of MemoryRecord sorted by relevance
        """
        if not self._model or not self._index or not self._conn:
            # Fall back to keyword search
            return self._keyword_search(query, top_k, category)

        # Embed query
        query_embedding = self._embed(query)
        if query_embedding is None:
            return self._keyword_search(query, top_k, category)

        # Search FAISS
        scores, indices = self._index.search(query_embedding, min(top_k * 2, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            if float(score) < min_score:
                continue

            row_id = self._id_map[idx]
            record = self._get_record(row_id)
            if record is None:
                continue

            # Filter by category
            if category and record.category != category:
                continue

            record.relevance_score = float(score)
            results.append(record)

            if len(results) >= top_k:
                break

        # Update access counts
        for r in results:
            self._conn.execute(
                'UPDATE long_term_memory SET access_count = access_count + 1, updated_at = ? WHERE id = ?',
                (time.time(), r.id)
            )
        if results:
            self._conn.commit()

        return results

    def _keyword_search(self, query: str, top_k: int,
                        category: Optional[str] = None) -> List[MemoryRecord]:
        """Fallback keyword search when FAISS is unavailable"""
        if not self._conn:
            return []

        words = query.lower().split()
        if not words:
            return []

        # Simple LIKE search
        conditions = ['content LIKE ?'] * len(words)
        params = ['%%%s%%' % w for w in words[:5]]  # Max 5 keywords

        if category:
            conditions.append('category = ?')
            params.append(category)

        sql = '''
            SELECT * FROM long_term_memory
            WHERE (%s) AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY access_count DESC, updated_at DESC
            LIMIT ?
        ''' % ' OR '.join(conditions)
        params.extend([time.time(), top_k])

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def recall_by_key(self, key: str) -> Optional[MemoryRecord]:
        """Exact key lookup"""
        if not self._conn:
            return None

        row = self._conn.execute(
            'SELECT * FROM long_term_memory WHERE key = ?', (key,)
        ).fetchone()

        if row:
            self._conn.execute(
                'UPDATE long_term_memory SET access_count = access_count + 1, updated_at = ? WHERE key = ?',
                (time.time(), key)
            )
            self._conn.commit()
            return self._row_to_record(row)
        return None

    def get_preferences(self) -> List[MemoryRecord]:
        """Get all user preferences"""
        if not self._conn:
            return []

        rows = self._conn.execute('''
            SELECT * FROM long_term_memory
            WHERE category = 'preference'
            AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY access_count DESC, updated_at DESC
        ''', (time.time(),)).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_patterns(self, limit: int = 10) -> List[MemoryRecord]:
        """Get frequently accessed behavior patterns"""
        if not self._conn:
            return []

        rows = self._conn.execute('''
            SELECT * FROM long_term_memory
            WHERE category = 'pattern'
            AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY access_count DESC
            LIMIT ?
        ''', (time.time(), limit)).fetchall()

        return [self._row_to_record(row) for row in rows]

    def learn_correction(self, utterance: str, correct_intent: str,
                         params: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Store a routing correction for future training.

        When the user corrects a misrouted intent, store it so:
        1. We can retrieve it for immediate routing improvement
        2. It feeds into the LoRA fine-tuning pipeline (Phase 6)
        """
        return self.store(
            key='correction:%s' % utterance[:100],
            content=utterance,
            category='correction',
            metadata={
                'correct_intent': correct_intent,
                'params': params or {},
                'timestamp': time.time(),
            },
        )

    def get_corrections(self, limit: int = 100) -> List[MemoryRecord]:
        """Get stored routing corrections for training"""
        if not self._conn:
            return []

        rows = self._conn.execute('''
            SELECT * FROM long_term_memory
            WHERE category = 'correction'
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,)).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_context_for_llm(self, query: str = "",
                            max_items: int = 5) -> str:
        """
        Build a context string for LLM injection.

        Combines relevant preferences, patterns, and knowledge.
        """
        parts = []

        # User preferences
        prefs = self.get_preferences()
        if prefs:
            pref_strs = ["%s: %s" % (p.key, p.content) for p in prefs[:3]]
            parts.append("User preferences: %s" % "; ".join(pref_strs))

        # Semantic recall if query provided
        if query:
            relevant = self.recall(query, top_k=3, min_score=0.4)
            if relevant:
                for r in relevant:
                    if r.category != 'preference':  # Don't duplicate prefs
                        parts.append("[%s] %s" % (r.category, r.content[:200]))

        # Common patterns
        patterns = self.get_patterns(limit=2)
        if patterns:
            pattern_strs = [p.content[:100] for p in patterns]
            parts.append("Common patterns: %s" % "; ".join(pattern_strs))

        return "\n".join(parts) if parts else ""

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove expired records"""
        if not self._conn:
            return 0

        now = time.time()
        cursor = self._conn.execute('''
            DELETE FROM long_term_memory
            WHERE expires_at IS NOT NULL AND expires_at < ?
        ''', (now,))
        self._conn.commit()

        removed = cursor.rowcount
        if removed > 0:
            self._rebuild_index()

        return removed

    def cleanup_stale(self, max_age_days: int = 90,
                      min_access_count: int = 2) -> int:
        """Remove old, rarely accessed records"""
        if not self._conn:
            return 0

        cutoff = time.time() - (max_age_days * 86400)
        cursor = self._conn.execute('''
            DELETE FROM long_term_memory
            WHERE updated_at < ? AND access_count < ?
            AND category NOT IN ('preference', 'correction')
        ''', (cutoff, min_access_count))
        self._conn.commit()

        removed = cursor.rowcount
        if removed > 0:
            self._rebuild_index()

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self._conn:
            return {'status': 'not initialized'}

        row = self._conn.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN category = 'preference' THEN 1 ELSE 0 END) as preferences,
                SUM(CASE WHEN category = 'pattern' THEN 1 ELSE 0 END) as patterns,
                SUM(CASE WHEN category = 'correction' THEN 1 ELSE 0 END) as corrections,
                SUM(CASE WHEN category = 'knowledge' THEN 1 ELSE 0 END) as knowledge
            FROM long_term_memory
            WHERE expires_at IS NULL OR expires_at > ?
        ''', (time.time(),)).fetchone()

        return {
            'total': row['total'] or 0,
            'preferences': row['preferences'] or 0,
            'patterns': row['patterns'] or 0,
            'corrections': row['corrections'] or 0,
            'knowledge': row['knowledge'] or 0,
            'faiss_vectors': self._index.ntotal if self._index else 0,
            'has_model': self._model is not None,
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _get_record(self, row_id: int) -> Optional[MemoryRecord]:
        """Get a record by SQLite row ID"""
        if not self._conn:
            return None

        row = self._conn.execute(
            'SELECT * FROM long_term_memory WHERE id = ?', (row_id,)
        ).fetchone()

        if not row:
            return None

        # Check expiry
        if row['expires_at'] and row['expires_at'] < time.time():
            return None

        return self._row_to_record(row)

    def _row_to_record(self, row) -> MemoryRecord:
        """Convert a SQLite row to MemoryRecord"""
        return MemoryRecord(
            id=row['id'],
            key=row['key'],
            content=row['content'],
            category=row['category'],
            metadata=json.loads(row['metadata_json']),
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            access_count=row['access_count'],
        )

    def shutdown(self) -> None:
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._is_ready = False
