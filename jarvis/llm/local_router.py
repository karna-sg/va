"""
Local LLM Router for Jarvis (Tier 2)

Uses Qwen3-4B via MLX for:
- Intent classification when Tier 1 (FAISS) is uncertain
- Slot extraction (entities, parameters)
- Compound command decomposition
- Determining if Claude is needed

Performance: ~2-3s on Apple Silicon via MLX (with prompt caching).

The model runs as an in-process MLX inference (no separate server needed).
Outputs structured JSON for deterministic routing.

Optimizations:
- Qwen3 thinking mode disabled (no <think> tags)
- System prompt KV cache pre-warmed at init
- Early stopping on complete JSON (brace counting)
- Compact system prompt to minimize tokens
"""

import copy
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from jarvis.intents.fast_router import RoutingResult


@dataclass
class Tier2Result:
    """Result from the local LLM router"""
    intent: str
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)
    needs_claude: bool = False
    missing_slots: List[str] = field(default_factory=list)
    reasoning: str = ""
    duration_ms: float = 0


# Compact system prompt - minimizes token count while preserving intent coverage.
# Uses shorthand notation: intent_name(param1,param2) and [needs_claude] markers.
ROUTER_SYSTEM_PROMPT = """Intent classifier for Kat, a voice assistant for developer Vasu. Output ONLY a single-line JSON object.

Intents: github.list_issues(repo,state), github.get_issue(repo,number), github.create_issue(repo,title,body), github.list_prs(repo,state), github.get_pr(repo,number)=show PR details, github.list_commits(repo,since), github.activity_yesterday(repo), github.activity_today(repo), github.activity_this_week(repo), github.repo_status(repo), slack.post_message(channel,message), slack.list_channels, slack.read_messages(channel), git.status, git.diff, git.branch, code.implement(needs_claude=true), code.fix_bug(needs_claude=true), code.review(needs_claude=true), code.explain(needs_claude=true), code.refactor(needs_claude=true), cli.run_tests, cli.run_build, workflow.daily_status(channel), workflow.pr_review(repo,number)=review/analyze a PR, workflow.sprint_planning(repo), meta.greeting, meta.thanks, meta.help, meta.cancel.

Format: {"intent":"x","confidence":0.9,"params":{},"needs_claude":false,"missing_slots":[]}
Unknown: {"intent":"unknown","confidence":0.0,"params":{},"needs_claude":true,"missing_slots":[]}"""


class LocalRouter:
    """
    Tier 2 intent router using local Qwen3-4B model via MLX.

    Used when Tier 1 (FAISS) confidence is below threshold.
    Provides structured intent classification with parameter extraction.

    Features:
    - In-process MLX inference (no server needed)
    - Structured JSON output with early stopping
    - Pre-warmed system prompt KV cache
    - Slot-filling detection (missing_slots)
    - Claude delegation decision (needs_claude)
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-4B-MLX-4bit",
                 adapter_path: Optional[str] = "~/.kat/adapters",
                 max_tokens: int = 96):
        self.model_name = model_name
        self.adapter_path = self._resolve_adapter(adapter_path)
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._is_ready = False
        self._system_cache = None
        self._system_token_count = 0

    @staticmethod
    def _resolve_adapter(adapter_path: Optional[str]) -> Optional[str]:
        """Resolve adapter path, returning None if not found"""
        if not adapter_path:
            return None
        import os
        path = os.path.expanduser(adapter_path)
        if os.path.exists(os.path.join(path, "adapters.safetensors")):
            return path
        return None

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def initialize(self) -> bool:
        """Load the MLX model (with LoRA adapter if available) and pre-warm cache"""
        try:
            from mlx_lm import load

            if self.adapter_path:
                print("  Loading local LLM: %s + LoRA adapter ..." % self.model_name)
            else:
                print("  Loading local LLM: %s ..." % self.model_name)
            start = time.time()

            self._model, self._tokenizer = load(
                self.model_name,
                adapter_path=self.adapter_path,
            )

            self._warm_system_cache()

            load_time = time.time() - start
            label = "fine-tuned" if self.adapter_path else "base"
            print("  Local LLM loaded (%s) in %.1fs (cache: %d tokens)" % (
                label, load_time, self._system_token_count))

            self._is_ready = True
            return True

        except ImportError:
            print("  mlx-lm not installed. Run: pip install mlx mlx-lm")
            return False
        except Exception as e:
            print("  Failed to load local LLM: %s" % e)
            return False

    def _warm_system_cache(self) -> None:
        """Pre-compute KV cache for the system prompt to speed up inference"""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        # Build system-only template tokens
        messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
        system_template = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        system_tokens = self._tokenizer.encode(system_template)
        self._system_token_count = len(system_tokens)

        # Create cache and prefill with system tokens
        self._system_cache = make_prompt_cache(self._model)
        self._model(
            mx.array(system_tokens)[None],  # add batch dimension
            cache=self._system_cache,
        )
        mx.eval([c.state for c in self._system_cache])

    def route(self, utterance: str,
              context: Optional[str] = None) -> Optional[Tier2Result]:
        """
        Classify an utterance into an intent using the local LLM.

        Args:
            utterance: User's voice command
            context: Optional conversation context

        Returns:
            Tier2Result with intent classification, or None on failure
        """
        if not self._is_ready:
            return None

        start = time.time()

        try:
            result = self._generate(utterance, context)
            duration = (time.time() - start) * 1000

            if result:
                result.duration_ms = duration

            return result

        except Exception as e:
            print("[Tier 2] Error: %s" % e)
            return None

    def _generate(self, utterance: str,
                  context: Optional[str] = None) -> Optional[Tier2Result]:
        """Generate intent classification via MLX inference with prompt caching"""
        import mlx.core as mx
        from mlx_lm import stream_generate

        # Build the user message
        user_msg = utterance
        if context:
            user_msg += "\nContext: %s" % context[:200]

        # Build full prompt tokens, then extract user-only portion
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        full_template = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        full_tokens = self._tokenizer.encode(full_template)
        user_only_tokens = full_tokens[self._system_token_count:]

        # Deep copy the pre-warmed cache so the original stays clean
        cached = copy.deepcopy(self._system_cache)

        # Stream generate with early stopping on complete JSON
        response = ""
        brace_depth = 0
        json_started = False

        for resp in stream_generate(
            self._model,
            self._tokenizer,
            prompt=mx.array(user_only_tokens),
            max_tokens=self.max_tokens,
            prompt_cache=cached,
        ):
            response += resp.text

            # Track brace depth for early stopping
            for ch in resp.text:
                if ch == '{':
                    brace_depth += 1
                    json_started = True
                elif ch == '}':
                    brace_depth -= 1
                    if json_started and brace_depth == 0:
                        return self._parse_response(response)

        # Fell through without early stop - try parsing anyway
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Optional[Tier2Result]:
        """Parse the LLM JSON response into a Tier2Result"""
        text = response.strip()

        # Handle markdown code fences
        if '```' in text:
            lines = text.split('\n')
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith('```'):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            text = '\n'.join(json_lines).strip()

        # Extract JSON object (model may output extra text before/after)
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            text = text[start:end + 1]

        try:
            data = json.loads(text)

            # Clean up intent name - model sometimes includes markers
            # like "code.fix_bug[needs_claude]" from the prompt notation
            intent = data.get('intent', 'unknown')
            needs_claude = bool(data.get('needs_claude', False))
            if '[needs_claude]' in intent:
                intent = intent.replace('[needs_claude]', '')
                needs_claude = True

            return Tier2Result(
                intent=intent,
                confidence=float(data.get('confidence', 0.0)),
                params=data.get('params', {}),
                needs_claude=needs_claude,
                missing_slots=data.get('missing_slots', []),
                reasoning=data.get('reasoning', ''),
            )

        except (json.JSONDecodeError, ValueError, KeyError):
            # Failed to parse - return None to fall through to Tier 3
            return None

    def to_routing_result(self, tier2: Tier2Result,
                          utterance: str) -> RoutingResult:
        """Convert a Tier2Result into a RoutingResult for the orchestrator"""
        intent_to_tool = {
            'github': 'github',
            'slack': 'slack',
            'git': 'local',
            'code': 'claude',
            'cli': 'local',
            'workflow': 'workflow',
            'meta': 'local',
        }

        prefix = tier2.intent.split('.')[0] if '.' in tier2.intent else ''
        tool = intent_to_tool.get(prefix, 'claude')

        needs_smart = tier2.needs_claude or tool in ('claude', 'workflow')
        model = 'smart' if needs_smart else 'fast'

        return RoutingResult(
            intent=tier2.intent,
            confidence=tier2.confidence,
            params=tier2.params,
            tool=tool,
            action=tier2.intent.split('.')[-1] if '.' in tier2.intent else '',
            model=model,
            response_template='',
            matched_phrase=utterance,
            tier=2,
            needs_claude=tier2.needs_claude,
        )

    def classify_batch(self, utterances: List[str]) -> List[Optional[Tier2Result]]:
        """Classify multiple utterances (useful for testing/benchmarking)"""
        results = []
        for utterance in utterances:
            results.append(self.route(utterance))
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            'model': self.model_name,
            'adapter': self.adapter_path,
            'fine_tuned': self.adapter_path is not None,
            'is_ready': self._is_ready,
            'max_tokens': self.max_tokens,
            'system_cache_tokens': self._system_token_count,
        }
