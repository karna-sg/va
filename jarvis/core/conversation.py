"""
Enhanced Conversation Management for Jarvis

Tracks conversation context with:
- Entity tracking (repos, issues, channels, people)
- Tool result history for reference resolution
- Turn history with metadata
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict, List


@dataclass
class Entity:
    """A tracked entity from conversation"""
    type: str  # 'repo', 'issue', 'pr', 'channel', 'person', etc.
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    mentioned_at: datetime = field(default_factory=datetime.now)


@dataclass
class ToolResult:
    """Result from a tool execution"""
    tool_name: str
    result: Any
    timestamp: datetime = field(default_factory=datetime.now)
    summary: str = ""


@dataclass
class ConversationTurn:
    """A single conversation turn"""
    user_input: str
    response: str
    timestamp: datetime = field(default_factory=datetime.now)
    entities: List[Entity] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    model_used: str = ""
    duration_ms: float = 0


class ConversationManager:
    """
    Enhanced conversation context manager.

    Replaces basic ConversationContext with:
    - Entity tracking across turns
    - Tool result history for "send this to Slack"
    - Full turn history
    - Session persistence via Claude CLI session ID
    """

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self._turns: List[ConversationTurn] = []
        self._entities: Dict[str, Entity] = {}
        self._tool_results: List[ToolResult] = []
        self._session_id: Optional[str] = None
        self._started_at: Optional[datetime] = None
        self._is_active: bool = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]):
        self._session_id = value

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def last_user_input(self) -> str:
        if self._turns:
            return self._turns[-1].user_input
        return ""

    @property
    def last_response(self) -> str:
        if self._turns:
            return self._turns[-1].response
        return ""

    @property
    def last_tool_result(self) -> Optional[ToolResult]:
        """Get the most recent tool result (for 'send this to Slack')"""
        if self._tool_results:
            return self._tool_results[-1]
        return None

    def start(self) -> None:
        """Start a new conversation"""
        self._started_at = datetime.now()
        self._is_active = True
        self._turns.clear()
        self._entities.clear()
        self._tool_results.clear()

    def add_turn(self, user_input: str, response: str,
                 model: str = "", duration_ms: float = 0) -> ConversationTurn:
        """Add a conversation turn"""
        turn = ConversationTurn(
            user_input=user_input,
            response=response,
            model_used=model,
            duration_ms=duration_ms,
        )

        # Extract and track entities
        entities = self._extract_entities(user_input, response)
        turn.entities = entities
        for entity in entities:
            self._entities["%s:%s" % (entity.type, entity.name)] = entity

        self._turns.append(turn)

        # Keep turns bounded
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

        return turn

    def add_tool_result(self, tool_name: str, result: Any, summary: str = "") -> None:
        """Track a tool execution result"""
        self._tool_results.append(ToolResult(
            tool_name=tool_name,
            result=result,
            summary=summary,
        ))
        # Keep bounded
        if len(self._tool_results) > 20:
            self._tool_results = self._tool_results[-20:]

    def get_entity(self, entity_type: str) -> Optional[Entity]:
        """Get the most recently mentioned entity of a type"""
        matches = [e for k, e in self._entities.items() if e.type == entity_type]
        if matches:
            return max(matches, key=lambda e: e.mentioned_at)
        return None

    def get_context_summary(self) -> str:
        """Get a brief context summary for LLM injection"""
        parts = []

        if self._turns:
            last = self._turns[-1]
            parts.append("Last exchange: User asked '%s'" % last.user_input[:100])

        active_entities = list(self._entities.values())[-5:]
        if active_entities:
            entity_strs = ["%s: %s" % (e.type, e.name) for e in active_entities]
            parts.append("Active entities: %s" % ", ".join(entity_strs))

        if self._tool_results:
            last_tool = self._tool_results[-1]
            parts.append("Last tool: %s" % last_tool.tool_name)

        return "; ".join(parts) if parts else ""

    def end(self) -> None:
        """End the conversation"""
        self._is_active = False

    def reset(self) -> None:
        """Full reset"""
        self._session_id = None
        self._turns.clear()
        self._entities.clear()
        self._tool_results.clear()
        self._started_at = None
        self._is_active = False

    def _extract_entities(self, user_input: str, response: str) -> List[Entity]:
        """Extract entities from conversation text"""
        entities = []
        combined = "%s %s" % (user_input, response)

        # GitHub issues: #123 or issue 123
        for match in re.finditer(r'(?:#|issue\s+)(\d+)', combined, re.IGNORECASE):
            entities.append(Entity(type='issue', name=match.group(1)))

        # PR references
        for match in re.finditer(r'(?:pr|pull request)\s*#?(\d+)', combined, re.IGNORECASE):
            entities.append(Entity(type='pr', name=match.group(1)))

        # Repository names (owner/repo pattern)
        for match in re.finditer(r'([\w-]+/[\w.-]+)', combined):
            entities.append(Entity(type='repo', name=match.group(1)))

        # Slack channels (#channel-name, but not issue numbers)
        for match in re.finditer(r'#([\w-]+)', combined):
            if not match.group(1).isdigit():
                entities.append(Entity(type='channel', name=match.group(1)))

        return entities
