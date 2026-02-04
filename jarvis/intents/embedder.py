"""
Intent Embedder for Jarvis

Creates and queries a FAISS index of intent phrase embeddings.
Uses sentence-transformers (all-MiniLM-L6-v2) for embedding.

Tier 1 routing: <1ms cosine similarity search after initial load.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class EmbeddingMatch:
    """A match from the FAISS index"""
    intent: str
    phrase: str
    score: float  # cosine similarity (0-1)
    metadata: Dict[str, Any]


class IntentEmbedder:
    """
    Manages intent embeddings and FAISS index.

    Workflow:
    1. Load catalog phrases
    2. Embed with sentence-transformers
    3. Build FAISS index
    4. At runtime: embed query -> search index -> return matches
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 index_dir: Optional[str] = None):
        self.model_name = model_name
        self.index_dir = Path(index_dir) if index_dir else (
            Path.home() / '.jarvis' / 'index'
        )
        self._model = None
        self._index = None
        self._intent_map: List[Dict[str, Any]] = []  # Maps index position -> intent info
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self) -> bool:
        """Load the sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            return True
        except ImportError:
            print("  sentence-transformers not installed. Run: pip install sentence-transformers")
            return False
        except Exception as e:
            print("  Failed to load embedding model: %s" % e)
            return False

    def build_index(self, intents: Dict[str, Any]) -> bool:
        """
        Build FAISS index from intent catalog.

        Args:
            intents: Dict of intent_name -> intent_config from catalog.yaml

        Returns:
            True if index was built successfully
        """
        if not self._model:
            if not self.load_model():
                return False

        try:
            import faiss
        except ImportError:
            print("  faiss-cpu not installed. Run: pip install faiss-cpu")
            return False

        phrases = []
        self._intent_map = []

        for intent_name, config in intents.items():
            for phrase in config.get('phrases', []):
                phrases.append(phrase)
                self._intent_map.append({
                    'intent': intent_name,
                    'phrase': phrase,
                    'params': config.get('params', {}),
                    'tool': config.get('tool', ''),
                    'action': config.get('action', ''),
                    'response_template': config.get('response_template', ''),
                    'model': config.get('model', 'smart'),
                    'description': config.get('description', ''),
                })

        if not phrases:
            print("  No phrases to index")
            return False

        # Embed all phrases
        print("  Embedding %d phrases..." % len(phrases))
        embeddings = self._model.encode(phrases, normalize_embeddings=True,
                                         show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS index (inner product = cosine similarity on normalized vectors)
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)

        print("  FAISS index built: %d vectors, %d dimensions" % (len(phrases), dimension))
        self._is_loaded = True
        return True

    def save_index(self) -> bool:
        """Save FAISS index and metadata to disk"""
        if not self._index or not self._intent_map:
            return False

        try:
            import faiss

            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self._index, str(self.index_dir / 'intents.faiss'))

            # Save intent map
            with open(self.index_dir / 'intent_map.pkl', 'wb') as f:
                pickle.dump(self._intent_map, f)

            print("  Index saved to %s" % self.index_dir)
            return True

        except Exception as e:
            print("  Failed to save index: %s" % e)
            return False

    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        faiss_path = self.index_dir / 'intents.faiss'
        map_path = self.index_dir / 'intent_map.pkl'

        if not faiss_path.exists() or not map_path.exists():
            return False

        try:
            import faiss

            self._index = faiss.read_index(str(faiss_path))

            with open(map_path, 'rb') as f:
                self._intent_map = pickle.load(f)

            self._is_loaded = True
            print("  Loaded index: %d vectors" % self._index.ntotal)
            return True

        except Exception as e:
            print("  Failed to load index: %s" % e)
            return False

    def search(self, query: str, top_k: int = 3) -> List[EmbeddingMatch]:
        """
        Search for matching intents.

        Args:
            query: User utterance
            top_k: Number of results to return

        Returns:
            List of EmbeddingMatch sorted by score (highest first)
        """
        if not self._is_loaded or not self._model or not self._index:
            return []

        # Embed the query
        query_embedding = self._model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search FAISS index
        scores, indices = self._index.search(query_embedding, top_k)

        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._intent_map):
                continue

            info = self._intent_map[idx]
            matches.append(EmbeddingMatch(
                intent=info['intent'],
                phrase=info['phrase'],
                score=float(score),
                metadata=info,
            ))

        return matches

    def get_best_match(self, query: str) -> Optional[EmbeddingMatch]:
        """Get the single best matching intent"""
        matches = self.search(query, top_k=1)
        return matches[0] if matches else None

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'model': self.model_name,
            'is_loaded': self._is_loaded,
            'num_vectors': self._index.ntotal if self._index else 0,
            'num_intents': len(set(m['intent'] for m in self._intent_map)),
            'index_dir': str(self.index_dir),
        }
