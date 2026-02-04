"""
State Machine Module for Jarvis

Handles:
- Agent state management
- State transitions
- Event handling
- Conversation context
"""

import asyncio
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from datetime import datetime


class AgentState(Enum):
    """Voice agent states"""

    # Phase 1 states (without wake word)
    IDLE = auto()           # Waiting for user input (keyboard trigger)
    LISTENING = auto()      # Recording user speech
    PROCESSING_STT = auto() # Converting speech to text
    PROCESSING_LLM = auto() # Claude is thinking
    SPEAKING = auto()       # TTS is playing response
    WAITING_FOLLOWUP = auto()  # Waiting for potential follow-up

    # Phase 2 states (with wake word)
    DORMANT = auto()        # Waiting for wake word
    ACTIVATED = auto()      # Wake word detected, ready to listen

    # Error states
    ERROR = auto()          # Error occurred, needs recovery


@dataclass
class ConversationContext:
    """Tracks conversation state and history"""
    session_id: Optional[str] = None
    turn_count: int = 0
    last_user_input: str = ""
    last_response: str = ""
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    is_active: bool = False

    def start(self) -> None:
        """Start a new conversation"""
        self.started_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = True
        self.turn_count = 0

    def add_turn(self, user_input: str, response: str) -> None:
        """Add a conversation turn"""
        self.last_user_input = user_input
        self.last_response = response
        self.last_activity = datetime.now()
        self.turn_count += 1

    def end(self) -> None:
        """End the conversation"""
        self.is_active = False

    def reset(self) -> None:
        """Reset conversation context"""
        self.session_id = None
        self.turn_count = 0
        self.last_user_input = ""
        self.last_response = ""
        self.started_at = None
        self.last_activity = None
        self.is_active = False


@dataclass
class StateEvent:
    """Event that triggers state changes"""
    name: str
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)


class StateManager:
    """
    Manages the voice agent state machine.

    State Transitions (Phase 1 - without wake word):

    IDLE -> LISTENING (on: start_listening)
    LISTENING -> PROCESSING_STT (on: recording_complete)
    PROCESSING_STT -> PROCESSING_LLM (on: transcription_complete)
    PROCESSING_LLM -> SPEAKING (on: response_ready)
    SPEAKING -> WAITING_FOLLOWUP (on: speech_complete)
    WAITING_FOLLOWUP -> LISTENING (on: follow_up_detected)
    WAITING_FOLLOWUP -> IDLE (on: timeout)
    * -> ERROR (on: error)
    ERROR -> IDLE (on: recovery)
    """

    # Valid state transitions - more flexible to handle various flows
    TRANSITIONS = {
        AgentState.IDLE: [AgentState.IDLE, AgentState.LISTENING, AgentState.PROCESSING_LLM, AgentState.ERROR],
        AgentState.LISTENING: [AgentState.LISTENING, AgentState.PROCESSING_STT, AgentState.PROCESSING_LLM, AgentState.SPEAKING, AgentState.IDLE, AgentState.ERROR],
        AgentState.PROCESSING_STT: [AgentState.PROCESSING_LLM, AgentState.SPEAKING, AgentState.IDLE, AgentState.ERROR],
        AgentState.PROCESSING_LLM: [AgentState.SPEAKING, AgentState.LISTENING, AgentState.IDLE, AgentState.ERROR],
        AgentState.SPEAKING: [AgentState.WAITING_FOLLOWUP, AgentState.LISTENING, AgentState.PROCESSING_LLM, AgentState.IDLE, AgentState.ERROR],
        AgentState.WAITING_FOLLOWUP: [AgentState.LISTENING, AgentState.IDLE, AgentState.ERROR],
        AgentState.ERROR: [AgentState.IDLE, AgentState.LISTENING],

        # Phase 2 transitions
        AgentState.DORMANT: [AgentState.ACTIVATED, AgentState.ERROR],
        AgentState.ACTIVATED: [AgentState.LISTENING, AgentState.DORMANT, AgentState.ERROR],
    }

    def __init__(self, initial_state: AgentState = AgentState.IDLE):
        """
        Initialize state manager.

        Args:
            initial_state: Starting state
        """
        self._state = initial_state
        self._previous_state: Optional[AgentState] = None
        self._context = ConversationContext()

        # State change callbacks
        self._on_state_change: list[Callable[[AgentState, AgentState], None]] = []
        self._state_handlers: dict[AgentState, Callable[[], None]] = {}

        # Event queue
        self._event_queue: asyncio.Queue[StateEvent] = asyncio.Queue()

        # Timestamps
        self._state_entered_at: datetime = datetime.now()
        self._history: list[tuple[AgentState, datetime]] = [(initial_state, datetime.now())]

    @property
    def state(self) -> AgentState:
        """Get current state"""
        return self._state

    @property
    def previous_state(self) -> Optional[AgentState]:
        """Get previous state"""
        return self._previous_state

    @property
    def context(self) -> ConversationContext:
        """Get conversation context"""
        return self._context

    @property
    def time_in_state(self) -> float:
        """Get time spent in current state (seconds)"""
        return (datetime.now() - self._state_entered_at).total_seconds()

    def can_transition_to(self, new_state: AgentState) -> bool:
        """Check if transition to new state is valid"""
        valid_targets = self.TRANSITIONS.get(self._state, [])
        return new_state in valid_targets

    def transition_to(self, new_state: AgentState, force: bool = False) -> bool:
        """
        Transition to a new state.

        Args:
            new_state: Target state
            force: If True, bypass transition validation

        Returns:
            True if transition succeeded
        """
        if not force and not self.can_transition_to(new_state):
            print(f"Invalid transition: {self._state.name} -> {new_state.name}")
            return False

        self._previous_state = self._state
        self._state = new_state
        self._state_entered_at = datetime.now()
        self._history.append((new_state, datetime.now()))

        # Keep history bounded
        if len(self._history) > 100:
            self._history = self._history[-50:]

        # Call state change callbacks
        for callback in self._on_state_change:
            try:
                callback(self._previous_state, new_state)
            except Exception as e:
                print(f"State change callback error: {e}")

        # Call state-specific handler
        if new_state in self._state_handlers:
            try:
                self._state_handlers[new_state]()
            except Exception as e:
                print(f"State handler error: {e}")

        return True

    def on_state_change(self, callback: Callable[[AgentState, AgentState], None]) -> None:
        """Register a state change callback"""
        self._on_state_change.append(callback)

    def set_state_handler(self, state: AgentState, handler: Callable[[], None]) -> None:
        """Set handler for when entering a specific state"""
        self._state_handlers[state] = handler

    async def emit_event(self, event_name: str, data: Any = None) -> None:
        """Emit an event to the state machine"""
        event = StateEvent(name=event_name, data=data)
        await self._event_queue.put(event)

    async def get_next_event(self, timeout: Optional[float] = None) -> Optional[StateEvent]:
        """Get the next event from the queue"""
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=timeout
                )
            return await self._event_queue.get()
        except asyncio.TimeoutError:
            return None

    def is_listening(self) -> bool:
        """Check if agent is in a listening state"""
        return self._state in [AgentState.LISTENING, AgentState.WAITING_FOLLOWUP]

    def is_processing(self) -> bool:
        """Check if agent is processing"""
        return self._state in [AgentState.PROCESSING_STT, AgentState.PROCESSING_LLM]

    def is_speaking(self) -> bool:
        """Check if agent is speaking"""
        return self._state == AgentState.SPEAKING

    def is_ready(self) -> bool:
        """Check if agent is ready for new input"""
        return self._state in [AgentState.IDLE, AgentState.WAITING_FOLLOWUP]

    def is_error(self) -> bool:
        """Check if agent is in error state"""
        return self._state == AgentState.ERROR

    def get_state_info(self) -> dict:
        """Get information about current state"""
        return {
            'state': self._state.name,
            'previous_state': self._previous_state.name if self._previous_state else None,
            'time_in_state': self.time_in_state,
            'conversation_active': self._context.is_active,
            'conversation_turns': self._context.turn_count,
        }

    def reset(self) -> None:
        """Reset state machine to initial state"""
        self._state = AgentState.IDLE
        self._previous_state = None
        self._context.reset()
        self._state_entered_at = datetime.now()

        # Clear event queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


# State transition helper functions
def create_phase1_state_machine() -> StateManager:
    """Create state machine configured for Phase 1 (no wake word)"""
    return StateManager(initial_state=AgentState.IDLE)


def create_phase2_state_machine() -> StateManager:
    """Create state machine configured for Phase 2 (with wake word)"""
    return StateManager(initial_state=AgentState.DORMANT)


# Test function
async def test_state_machine():
    """Test state machine functionality"""
    print("Testing State Machine...")

    sm = create_phase1_state_machine()

    # Register callback
    def on_change(old_state, new_state):
        print(f"  State changed: {old_state.name} -> {new_state.name}")

    sm.on_state_change(on_change)

    # Test transitions
    print("\nTesting Phase 1 transitions:")

    # IDLE -> LISTENING
    assert sm.transition_to(AgentState.LISTENING)
    assert sm.state == AgentState.LISTENING

    # LISTENING -> PROCESSING_STT
    assert sm.transition_to(AgentState.PROCESSING_STT)

    # PROCESSING_STT -> PROCESSING_LLM
    assert sm.transition_to(AgentState.PROCESSING_LLM)

    # PROCESSING_LLM -> SPEAKING
    assert sm.transition_to(AgentState.SPEAKING)

    # SPEAKING -> WAITING_FOLLOWUP
    assert sm.transition_to(AgentState.WAITING_FOLLOWUP)

    # WAITING_FOLLOWUP -> IDLE (timeout)
    assert sm.transition_to(AgentState.IDLE)

    # Test invalid transition
    print("\nTesting invalid transition (IDLE -> SPEAKING):")
    assert not sm.transition_to(AgentState.SPEAKING)

    # Test error and recovery
    print("\nTesting error and recovery:")
    assert sm.transition_to(AgentState.ERROR)
    assert sm.is_error()
    assert sm.transition_to(AgentState.IDLE)

    # Test state info
    print(f"\nState info: {sm.get_state_info()}")

    # Test context
    print("\nTesting conversation context:")
    sm.context.start()
    sm.context.add_turn("Hello", "Hi there!")
    sm.context.add_turn("How are you?", "I'm doing well!")
    print(f"  Turns: {sm.context.turn_count}")
    print(f"  Last input: {sm.context.last_user_input}")

    print("\nState machine test complete!")


if __name__ == "__main__":
    asyncio.run(test_state_machine())
