"""
Training Data Collector for Jarvis

Extracts training data from conversation logs for:
1. Intent routing improvements (utterance -> intent mappings)
2. Preference extraction (user habits and settings)
3. LoRA fine-tuning dataset preparation (Phase 6)

Data flow:
  Conversation logs -> extract successful routings -> augment with paraphrases
  -> format for training -> output JSONL for fine-tuning

Usage:
  python -m jarvis.training.collect [--export training_data.jsonl] [--stats]
"""

import json
import time
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict


@dataclass
class TrainingSample:
    """A single training sample for intent routing"""
    utterance: str
    intent: str
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ''  # 'correction', 'confirmed', 'conversation', 'augmented'
    timestamp: float = 0


@dataclass
class PreferenceExtraction:
    """An extracted user preference"""
    key: str
    value: str
    context: str  # The conversation turn that revealed this
    confidence: float = 1.0


class TrainingDataCollector:
    """
    Collects and manages training data from conversation logs.

    Sources:
    1. Routing corrections (user corrected a misrouted intent)
    2. Confirmed routings (Tier 1 match that user accepted)
    3. Conversation analysis (extract intent/params from Claude responses)

    Storage: SQLite at ~/.jarvis/training.db
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else (
            Path.home() / '.jarvis' / 'training.db'
        )
        self._conn = None  # type: Optional[sqlite3.Connection]

    def initialize(self) -> bool:
        """Initialize the training database"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

            self._conn.executescript('''
                CREATE TABLE IF NOT EXISTS training_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    utterance TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL DEFAULT 'conversation',
                    created_at REAL NOT NULL,
                    is_validated INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS conversation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_input TEXT NOT NULL,
                    response TEXT NOT NULL,
                    intent TEXT,
                    tier INTEGER,
                    confidence REAL,
                    duration_ms REAL,
                    model_used TEXT,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS preference_extractions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    context TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    is_applied INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_ts_intent ON training_samples(intent);
                CREATE INDEX IF NOT EXISTS idx_ts_source ON training_samples(source);
                CREATE INDEX IF NOT EXISTS idx_cl_session ON conversation_logs(session_id);
                CREATE INDEX IF NOT EXISTS idx_cl_intent ON conversation_logs(intent);
            ''')
            self._conn.commit()
            return True

        except Exception as e:
            print("  Training collector error: %s" % e)
            return False

    # -------------------------------------------------------------------------
    # Data Collection
    # -------------------------------------------------------------------------

    def log_conversation(self, user_input: str, response: str,
                         session_id: Optional[str] = None,
                         intent: Optional[str] = None,
                         tier: Optional[int] = None,
                         confidence: Optional[float] = None,
                         duration_ms: float = 0,
                         model_used: str = '') -> Optional[int]:
        """
        Log a conversation turn for later analysis.

        Called by the orchestrator after each turn.
        """
        if not self._conn:
            return None

        cursor = self._conn.execute('''
            INSERT INTO conversation_logs
            (session_id, user_input, response, intent, tier, confidence,
             duration_ms, model_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_input, response[:2000], intent, tier,
              confidence, duration_ms, model_used, time.time()))
        self._conn.commit()
        return cursor.lastrowid

    def record_confirmed_routing(self, utterance: str, intent: str,
                                 params: Optional[Dict[str, Any]] = None,
                                 confidence: float = 1.0) -> Optional[int]:
        """
        Record a confirmed routing (user accepted the Tier 1 match).

        These are high-quality training samples since the user didn't correct them.
        """
        if not self._conn:
            return None

        cursor = self._conn.execute('''
            INSERT INTO training_samples
            (utterance, intent, params_json, confidence, source, created_at, is_validated)
            VALUES (?, ?, ?, ?, 'confirmed', ?, 1)
        ''', (utterance, intent, json.dumps(params or {}),
              confidence, time.time()))
        self._conn.commit()
        return cursor.lastrowid

    def record_correction(self, utterance: str, wrong_intent: str,
                          correct_intent: str,
                          params: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Record a routing correction.

        The user corrected a misrouted intent - these are valuable for training.
        """
        if not self._conn:
            return None

        # Store the correction as a training sample
        cursor = self._conn.execute('''
            INSERT INTO training_samples
            (utterance, intent, params_json, confidence, source, created_at, is_validated)
            VALUES (?, ?, ?, 1.0, 'correction', ?, 1)
        ''', (utterance, correct_intent, json.dumps(params or {}), time.time()))
        self._conn.commit()

        return cursor.lastrowid

    def extract_preference(self, key: str, value: str,
                           context: str = '',
                           confidence: float = 1.0) -> Optional[int]:
        """
        Record an extracted user preference.

        Preferences are extracted from conversation patterns, e.g.:
        - "I always use the develop branch" -> pref.default_branch = develop
        - "Post to #engineering" -> pref.default_channel = #engineering
        """
        if not self._conn:
            return None

        cursor = self._conn.execute('''
            INSERT INTO preference_extractions
            (key, value, context, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (key, value, context[:500], confidence, time.time()))
        self._conn.commit()
        return cursor.lastrowid

    # -------------------------------------------------------------------------
    # Data Export
    # -------------------------------------------------------------------------

    def export_training_data(self, output_path: Optional[str] = None,
                             min_confidence: float = 0.8,
                             validated_only: bool = False) -> List[TrainingSample]:
        """
        Export training samples as JSONL (for LoRA fine-tuning).

        Args:
            output_path: File path for JSONL output (None = return list only)
            min_confidence: Minimum confidence threshold
            validated_only: Only include validated samples

        Returns:
            List of TrainingSample objects
        """
        if not self._conn:
            return []

        conditions = ['confidence >= ?']
        params_list = [min_confidence]  # type: List[Any]

        if validated_only:
            conditions.append('is_validated = 1')

        sql = '''
            SELECT utterance, intent, params_json, confidence, source, created_at
            FROM training_samples
            WHERE %s
            ORDER BY created_at
        ''' % ' AND '.join(conditions)

        rows = self._conn.execute(sql, params_list).fetchall()

        samples = []
        for row in rows:
            sample = TrainingSample(
                utterance=row['utterance'],
                intent=row['intent'],
                params=json.loads(row['params_json']),
                confidence=row['confidence'],
                source=row['source'],
                timestamp=row['created_at'],
            )
            samples.append(sample)

        # Write to JSONL if path provided
        if output_path and samples:
            with open(output_path, 'w') as f:
                for sample in samples:
                    # Format for fine-tuning: instruction-response pairs
                    record = {
                        'instruction': 'Classify this voice command into an intent with parameters.',
                        'input': sample.utterance,
                        'output': json.dumps({
                            'intent': sample.intent,
                            'params': sample.params,
                            'confidence': sample.confidence,
                        }),
                        'source': sample.source,
                    }
                    f.write(json.dumps(record) + '\n')

            print("Exported %d samples to %s" % (len(samples), output_path))

        return samples

    def export_preferences(self) -> List[PreferenceExtraction]:
        """Export extracted preferences"""
        if not self._conn:
            return []

        rows = self._conn.execute('''
            SELECT key, value, context, confidence FROM preference_extractions
            WHERE is_applied = 0
            ORDER BY confidence DESC, created_at DESC
        ''').fetchall()

        return [
            PreferenceExtraction(
                key=row['key'],
                value=row['value'],
                context=row['context'],
                confidence=row['confidence'],
            )
            for row in rows
        ]

    def mark_preference_applied(self, key: str) -> None:
        """Mark a preference extraction as applied"""
        if not self._conn:
            return

        self._conn.execute(
            'UPDATE preference_extractions SET is_applied = 1 WHERE key = ?',
            (key,)
        )
        self._conn.commit()

    # -------------------------------------------------------------------------
    # Analytics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get training data statistics"""
        if not self._conn:
            return {'status': 'not initialized'}

        samples_row = self._conn.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN source = 'confirmed' THEN 1 ELSE 0 END) as confirmed,
                SUM(CASE WHEN source = 'correction' THEN 1 ELSE 0 END) as corrections,
                SUM(CASE WHEN source = 'augmented' THEN 1 ELSE 0 END) as augmented,
                SUM(CASE WHEN is_validated = 1 THEN 1 ELSE 0 END) as validated
            FROM training_samples
        ''').fetchone()
        # SQLite SUM returns None for empty tables
        samples = {
            'total': samples_row['total'] or 0,
            'confirmed': samples_row['confirmed'] or 0,
            'corrections': samples_row['corrections'] or 0,
            'augmented': samples_row['augmented'] or 0,
            'validated': samples_row['validated'] or 0,
        }

        logs = self._conn.execute(
            'SELECT COUNT(*) as total FROM conversation_logs'
        ).fetchone()

        prefs = self._conn.execute(
            'SELECT COUNT(*) as total FROM preference_extractions'
        ).fetchone()

        # Intent distribution
        intent_dist = self._conn.execute('''
            SELECT intent, COUNT(*) as count
            FROM training_samples
            GROUP BY intent
            ORDER BY count DESC
            LIMIT 10
        ''').fetchall()

        return {
            'training_samples': samples,
            'conversation_logs': logs['total'],
            'preference_extractions': prefs['total'],
            'top_intents': [
                {'intent': row['intent'], 'count': row['count']}
                for row in intent_dist
            ],
        }

    def get_routing_accuracy(self) -> Dict[str, Any]:
        """
        Estimate routing accuracy from corrections vs confirmed.

        A high correction rate indicates the router needs improvement.
        """
        if not self._conn:
            return {}

        row = self._conn.execute('''
            SELECT
                SUM(CASE WHEN source = 'confirmed' THEN 1 ELSE 0 END) as confirmed,
                SUM(CASE WHEN source = 'correction' THEN 1 ELSE 0 END) as corrections
            FROM training_samples
        ''').fetchone()

        confirmed = row['confirmed'] or 0
        corrections = row['corrections'] or 0
        total = confirmed + corrections

        accuracy = (confirmed / total * 100) if total > 0 else 0

        return {
            'total_routed': total,
            'confirmed': confirmed,
            'corrections': corrections,
            'accuracy_pct': round(accuracy, 1),
        }

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def cleanup(self, max_age_days: int = 90) -> int:
        """Remove old unvalidated samples and old logs"""
        if not self._conn:
            return 0

        cutoff = time.time() - (max_age_days * 86400)

        # Remove old unvalidated samples
        c1 = self._conn.execute('''
            DELETE FROM training_samples
            WHERE created_at < ? AND is_validated = 0
        ''', (cutoff,))

        # Remove old conversation logs
        c2 = self._conn.execute('''
            DELETE FROM conversation_logs
            WHERE created_at < ?
        ''', (cutoff,))

        self._conn.commit()
        return c1.rowcount + c2.rowcount

    def shutdown(self) -> None:
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    """CLI for training data management"""
    import sys

    collector = TrainingDataCollector()
    if not collector.initialize():
        print("Failed to initialize training collector")
        sys.exit(1)

    if '--stats' in sys.argv:
        stats = collector.get_stats()
        print("Training Data Statistics:")
        print("  Samples: %d total (%d confirmed, %d corrections)" % (
            stats['training_samples']['total'],
            stats['training_samples']['confirmed'],
            stats['training_samples']['corrections'],
        ))
        print("  Conversation logs: %d" % stats['conversation_logs'])
        print("  Preference extractions: %d" % stats['preference_extractions'])

        accuracy = collector.get_routing_accuracy()
        if accuracy.get('total_routed', 0) > 0:
            print("\n  Routing accuracy: %.1f%% (%d/%d)" % (
                accuracy['accuracy_pct'],
                accuracy['confirmed'],
                accuracy['total_routed'],
            ))

        if stats['top_intents']:
            print("\n  Top intents:")
            for item in stats['top_intents']:
                print("    %s: %d samples" % (item['intent'], item['count']))

    elif '--export' in sys.argv:
        idx = sys.argv.index('--export')
        output = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'training_data.jsonl'
        samples = collector.export_training_data(output_path=output)
        print("Exported %d training samples" % len(samples))

    elif '--accuracy' in sys.argv:
        accuracy = collector.get_routing_accuracy()
        if accuracy.get('total_routed', 0) > 0:
            print("Routing accuracy: %.1f%%" % accuracy['accuracy_pct'])
            print("  Confirmed: %d" % accuracy['confirmed'])
            print("  Corrections: %d" % accuracy['corrections'])
        else:
            print("No routing data collected yet")

    else:
        print("Usage: python -m jarvis.training.collect [--stats] [--export FILE] [--accuracy]")

    collector.shutdown()


if __name__ == '__main__':
    main()
