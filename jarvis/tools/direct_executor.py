"""
Direct Executor for Jarvis

Executes Tier 1 matched intents with minimal overhead.

For intents that don't need Claude's reasoning:
- Meta intents (greeting, thanks, help) -> instant local response
- Simple tool calls -> targeted Claude CLI prompt (fast model)

This saves 500-2000ms vs full conversational Claude flow because:
1. We already know the intent (no LLM decision-making needed)
2. We use haiku (fast model) with a minimal targeted prompt
3. We skip the full system prompt for direct tool calls
"""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from jarvis.intents.fast_router import RoutingResult


@dataclass
class ExecutionResult:
    """Result of direct execution"""
    text: str
    spoken_text: str  # TTS-friendly version
    is_error: bool = False
    duration_ms: float = 0
    source: str = ""  # 'local', 'claude_fast', 'claude_smart'


# Local responses that don't need any LLM or tool call
LOCAL_RESPONSES = {
    'meta.greeting': [
        "Hey! What can I help with?",
        "Hi there! What do you need?",
        "Hello! Ready to help.",
    ],
    'meta.thanks': [
        "Anytime!",
        "You're welcome!",
        "Happy to help!",
    ],
    'meta.help': [
        "I can help with GitHub issues, PRs, commits, Slack messaging, "
        "git operations, code review, and more. Just ask!",
    ],
    'meta.cancel': [
        "Cancelled.",
        "Alright, cancelled.",
    ],
}

# Prompt templates for tool-based intents (minimal, targeted)
TOOL_PROMPTS = {
    'github.list_issues': (
        "List open issues on {owner}/{repo}. "
        "Give me a count and the top 3 by priority. Keep it to 2 sentences."
    ),
    'github.get_issue': (
        "Show issue #{number} on {owner}/{repo}. "
        "Give me the title, status, and a one-line summary."
    ),
    'github.list_prs': (
        "List open pull requests on {owner}/{repo}. "
        "Give me a count and the most recent one. Keep it to 2 sentences."
    ),
    'github.get_pr': (
        "Show PR #{number} on {owner}/{repo}. "
        "Give me the title, status, and a one-line summary."
    ),
    'github.list_commits': (
        "Show recent commits on {owner}/{repo}. "
        "List the last 5 with short messages. Keep it brief."
    ),
    'github.activity_yesterday': (
        "What work was done yesterday on {owner}/{repo}? "
        "Summarize commits and any PRs merged. Keep it to 2-3 sentences."
    ),
    'github.activity_today': (
        "What work was done today on {owner}/{repo}? "
        "Summarize commits and any PRs merged. Keep it to 2-3 sentences."
    ),
    'github.activity_this_week': (
        "Summarize this week's activity on {owner}/{repo}. "
        "Commits, PRs, issues. Keep it to 3 sentences."
    ),
    'github.repo_status': (
        "Give a brief status of {owner}/{repo}. "
        "Open issues, open PRs, recent activity. 2-3 sentences."
    ),
    'slack.list_channels': (
        "List the Slack channels. Give me a count and the most active ones."
    ),
    'slack.read_messages': (
        "Show recent messages from Slack channel {channel}. "
        "Summarize the last few messages briefly."
    ),
    'git.status': (
        "Run git status and tell me what's changed. Keep it brief."
    ),
    'git.diff': (
        "Show a brief summary of the current git diff. What files changed and why?"
    ),
    'git.branch': (
        "What git branch am I on? Just the branch name."
    ),
}


class DirectExecutor:
    """
    Executes matched intents with minimal latency.

    Routes to either:
    1. Local response (no LLM needed) - ~0ms
    2. Targeted Claude CLI call (fast model) - saves 500ms+ vs full flow
    3. Full Claude CLI (for complex intents) - falls through to orchestrator
    """

    def __init__(self, claude=None, defaults: Optional[Dict[str, str]] = None):
        """
        Args:
            claude: ClaudeCode instance for tool-based calls
            defaults: Default parameter values (owner, repo, channel)
        """
        self.claude = claude
        self.defaults = defaults or {}
        self._response_index = 0  # For cycling through local responses
        self._has_session = False  # Track if we have a reusable Claude session

    async def execute(self, routing: RoutingResult,
                      utterance: str = "") -> Optional[ExecutionResult]:
        """
        Execute a routed intent.

        Args:
            routing: RoutingResult from FastRouter
            utterance: Original user utterance (for context)

        Returns:
            ExecutionResult if handled, None if should fall through to Claude
        """
        start = time.time()

        # 1. Local responses (instant, no LLM)
        if routing.tool == 'local' and routing.intent in LOCAL_RESPONSES:
            responses = LOCAL_RESPONSES[routing.intent]
            text = responses[self._response_index % len(responses)]
            self._response_index += 1
            return ExecutionResult(
                text=text,
                spoken_text=text,
                duration_ms=(time.time() - start) * 1000,
                source='local',
            )

        # 2. Targeted tool calls via Claude CLI (fast model, minimal prompt)
        if routing.intent in TOOL_PROMPTS and self.claude:
            prompt = self._build_prompt(routing, utterance)
            if prompt:
                return await self._execute_with_claude(
                    prompt, routing, start)

        # 3. Complex intents or workflows -> fall through to orchestrator
        return None

    def _build_prompt(self, routing: RoutingResult, utterance: str) -> Optional[str]:
        """Build a targeted prompt for the matched intent"""
        template = TOOL_PROMPTS.get(routing.intent)
        if not template:
            return None

        # Merge defaults with extracted params
        params = dict(self.defaults)
        params.update(routing.params)

        # Ensure owner is set
        if 'owner' not in params:
            params['owner'] = self.defaults.get('default_owner', 'karna-sg')
        if 'repo' not in params:
            params['repo'] = self.defaults.get('default_repo', 'curiescious')
        if 'channel' not in params:
            params['channel'] = self.defaults.get('default_channel', '#general')

        try:
            return template.format(**params)
        except KeyError:
            # Missing param, pass the utterance through instead
            return utterance

    async def _execute_with_claude(self, prompt: str,
                                   routing: RoutingResult,
                                   start_time: float) -> ExecutionResult:
        """Execute a targeted prompt via Claude CLI"""
        try:
            # Force the model based on routing (bypass auto-classification
            # which would always pick 'smart' due to tool keywords in prompt)
            model = 'haiku' if routing.model == 'fast' else 'sonnet'

            # Reuse session for faster responses (skip MCP re-discovery)
            response = await self.claude.send(
                prompt,
                new_conversation=(not self._has_session),
                model=model,
            )
            self._has_session = True

            duration = (time.time() - start_time) * 1000

            if response.is_error:
                return ExecutionResult(
                    text="Sorry, I couldn't get that information.",
                    spoken_text="Sorry, I couldn't get that information.",
                    is_error=True,
                    duration_ms=duration,
                    source='claude_fast',
                )

            return ExecutionResult(
                text=response.text,
                spoken_text=response.text,
                duration_ms=duration,
                source='claude_fast',
            )

        except Exception as e:
            return ExecutionResult(
                text="Error: %s" % str(e),
                spoken_text="Sorry, something went wrong.",
                is_error=True,
                duration_ms=(time.time() - start_time) * 1000,
                source='claude_fast',
            )

    def can_execute_locally(self, intent: str) -> bool:
        """Check if an intent can be executed without any LLM call"""
        return intent in LOCAL_RESPONSES
