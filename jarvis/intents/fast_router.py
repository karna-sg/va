"""
Fast Router for Jarvis (Tier 1)

Routes user utterances to intents using FAISS embedding search.
If confidence > threshold, executes directly without LLM.

Performance: <1ms search after model load (~5ms embed + <1ms FAISS).
"""

import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from jarvis.intents.embedder import IntentEmbedder, EmbeddingMatch


@dataclass
class RoutingResult:
    """Result of intent routing"""
    intent: str
    confidence: float
    params: Dict[str, Any]
    tool: str
    action: str
    model: str  # 'fast' or 'smart'
    response_template: str
    matched_phrase: str
    tier: int  # 1 = FAISS, 2 = local LLM, 3 = Claude
    needs_claude: bool  # Whether this must go through Claude


class FastRouter:
    """
    Tier 1 intent router using FAISS embedding search.

    Flow:
    1. Embed user utterance (~5ms)
    2. FAISS cosine similarity search (<1ms)
    3. If top match > threshold (0.82), route directly
    4. Otherwise, fall through to Tier 2/3

    Also handles:
    - Parameter extraction from utterance (e.g., issue numbers)
    - Default parameter resolution
    - Intent-specific model selection
    """

    def __init__(self, catalog_path: Optional[str] = None,
                 index_dir: Optional[str] = None):
        self._catalog_path = catalog_path or str(
            Path(__file__).parent / 'catalog.yaml'
        )
        self._catalog: Dict[str, Any] = {}
        self._settings: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}
        self._embedder = IntentEmbedder(index_dir=index_dir)
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def load_catalog(self) -> bool:
        """Load the intent catalog from YAML"""
        try:
            with open(self._catalog_path) as f:
                data = yaml.safe_load(f)

            self._catalog = data.get('intents', {})
            self._settings = data.get('settings', {})
            self._defaults = data.get('defaults', {})

            print("  Intent catalog: %d intents loaded" % len(self._catalog))
            return True

        except Exception as e:
            print("  Failed to load catalog: %s" % e)
            return False

    async def initialize(self) -> bool:
        """
        Initialize the fast router.

        Tries to load pre-built index first, falls back to building new one.
        """
        # Load catalog
        if not self.load_catalog():
            return False

        # Try loading pre-built index
        if self._embedder.load_index():
            # Still need the model for runtime queries
            if self._embedder.load_model():
                self._is_ready = True
                return True

        # Build new index
        print("  Building FAISS index...")
        if not self._embedder.build_index(self._catalog):
            return False

        # Save for next time
        self._embedder.save_index()

        self._is_ready = True
        return True

    # Greeting prefixes to strip from compound utterances
    GREETING_PREFIXES = [
        'hey jarvis ', 'hi jarvis ', 'hello jarvis ',
        'hey jarvis, ', 'hi jarvis, ', 'hello jarvis, ',
        'jarvis ', 'jarvis, ',
    ]

    def _strip_greeting_prefix(self, utterance: str):
        """Strip greeting prefix from compound utterances like 'hey jarvis show me issues'.
        Returns (cleaned_utterance, had_greeting)."""
        lower = utterance.lower()
        for prefix in self.GREETING_PREFIXES:
            if lower.startswith(prefix):
                remainder = utterance[len(prefix):].strip()
                if remainder:  # Only strip if there's content after the greeting
                    return remainder, True
        return utterance, False

    def route(self, utterance: str) -> Optional[RoutingResult]:
        """
        Route an utterance to an intent.

        Args:
            utterance: User's speech-to-text output

        Returns:
            RoutingResult if confident match found, None if should fall through
        """
        if not self._is_ready:
            return None

        threshold = self._settings.get('tier1_threshold', 0.78)

        # Strip greeting prefix for compound utterances
        # "hey jarvis show me the issues" -> search "show me the issues"
        cleaned, had_greeting = self._strip_greeting_prefix(utterance)

        # If utterance was ONLY a greeting (nothing left), use original
        if had_greeting and not cleaned.strip():
            cleaned = utterance

        # Search with cleaned utterance first
        match = self._embedder.get_best_match(cleaned)

        # If no match with cleaned text, try original as fallback
        if (not match or match.score < threshold) and cleaned != utterance:
            match = self._embedder.get_best_match(utterance)

        if not match or match.score < threshold:
            return None

        # Extract parameters from utterance
        params = self._resolve_params(
            match.metadata.get('params', {}),
            utterance
        )

        # Determine if this needs Claude
        tool = match.metadata.get('tool', '')
        needs_claude = tool in ('claude', 'workflow') or match.metadata.get('model') == 'smart'

        return RoutingResult(
            intent=match.intent,
            confidence=match.score,
            params=params,
            tool=tool,
            action=match.metadata.get('action', ''),
            model=match.metadata.get('model', 'smart'),
            response_template=match.metadata.get('response_template', ''),
            matched_phrase=match.phrase,
            tier=1,
            needs_claude=needs_claude,
        )

    def route_with_details(self, utterance: str, top_k: int = 3) -> Tuple[Optional[RoutingResult], list]:
        """
        Route with full debug details.

        Returns:
            Tuple of (best RoutingResult or None, list of top-K matches)
        """
        if not self._is_ready:
            return None, []

        threshold = self._settings.get('tier1_threshold', 0.78)
        matches = self._embedder.search(utterance, top_k=top_k)

        if not matches or matches[0].score < threshold:
            return None, matches

        best = matches[0]
        params = self._resolve_params(
            best.metadata.get('params', {}),
            utterance
        )

        tool = best.metadata.get('tool', '')
        needs_claude = tool in ('claude', 'workflow') or best.metadata.get('model') == 'smart'

        result = RoutingResult(
            intent=best.intent,
            confidence=best.score,
            params=params,
            tool=tool,
            action=best.metadata.get('action', ''),
            model=best.metadata.get('model', 'smart'),
            response_template=best.metadata.get('response_template', ''),
            matched_phrase=best.phrase,
            tier=1,
            needs_claude=needs_claude,
        )

        return result, matches

    def _resolve_params(self, param_template: Dict[str, Any],
                        utterance: str) -> Dict[str, Any]:
        """
        Resolve parameters from template + utterance.

        Handles:
        - ${default_repo} -> resolved from defaults
        - {number} -> extracted from utterance
        """
        params = {}

        for key, value in param_template.items():
            if isinstance(value, str):
                # Resolve ${variable} references
                if value.startswith('${') and value.endswith('}'):
                    var_name = value[2:-1]
                    params[key] = self._defaults.get(var_name, value)
                else:
                    params[key] = value
            else:
                params[key] = value

        # Extract numbers from utterance (issue #5, PR 12, etc.)
        numbers = re.findall(r'(?:#|number\s+|issue\s+|pr\s+)(\d+)', utterance.lower())
        if numbers:
            params['number'] = numbers[0]
        elif not params.get('number'):
            # Try bare numbers at the end
            bare = re.findall(r'\b(\d+)\b', utterance)
            if bare:
                params['number'] = bare[-1]

        return params

    def get_catalog_info(self) -> Dict[str, Any]:
        """Get information about loaded catalog"""
        total_phrases = sum(
            len(c.get('phrases', []))
            for c in self._catalog.values()
        )
        return {
            'num_intents': len(self._catalog),
            'total_phrases': total_phrases,
            'threshold': self._settings.get('tier1_threshold', 0.78),
            'embedding_model': self._settings.get('embedding_model', 'all-MiniLM-L6-v2'),
            'is_ready': self._is_ready,
        }
