"""
Claude Code CLI Integration Module for Jarvis

Handles:
- Programmatic invocation of Claude Code CLI
- Session management for multi-turn conversations
- JSON output parsing
- Streaming response support
- Permission handling
"""

import asyncio
import json
import time
from typing import Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum


class PermissionMode(Enum):
    """Claude Code permission modes"""
    DEFAULT = "default"
    BYPASS = "bypassPermissions"
    ACCEPT_EDITS = "acceptEdits"
    DONT_ASK = "dontAsk"


@dataclass
class ClaudeResponse:
    """Response from Claude Code CLI"""
    text: str
    session_id: str
    is_error: bool = False
    duration_ms: float = 0
    cost_usd: float = 0
    model: str = ""
    num_turns: int = 0
    raw_response: dict = field(default_factory=dict)

    @property
    def is_asking_question(self) -> bool:
        """Check if the response is asking the user a question"""
        if not self.text:
            return False
        text_lower = self.text.lower().strip()
        # Check for question marks
        if '?' in self.text:
            return True
        # Check for common question patterns
        question_patterns = [
            'which repo', 'which repository', 'which project',
            'which file', 'which issue', 'which one',
            'could you', 'can you', 'would you',
            'do you want', 'should i', 'shall i',
            'let me know', 'please specify', 'please provide',
            'what would you like', 'how would you like',
            'is that correct', 'does that sound',
        ]
        return any(pattern in text_lower for pattern in question_patterns)

    @property
    def is_working(self) -> bool:
        """Check if Claude indicates it's actively working on something"""
        if not self.text:
            return False
        text_lower = self.text.lower()
        working_patterns = [
            'let me', "i'll", "i will", "i'm going to",
            'checking', 'looking', 'fetching', 'reading',
            'implementing', 'creating', 'updating', 'modifying',
        ]
        return any(pattern in text_lower for pattern in working_patterns)


@dataclass
class StreamChunk:
    """A chunk from streaming response"""
    text: str
    chunk_type: str  # 'text', 'tool_use', 'result', etc.
    is_final: bool = False


class ClaudeCode:
    """
    Claude Code CLI wrapper for voice agent integration.

    Supports:
    - Non-interactive mode with JSON output
    - Session resumption for multi-turn conversations
    - Streaming responses for real-time TTS
    - Permission bypass for automation
    - Project directory access for code changes
    - Model routing (fast vs smart models)
    """

    # Keywords that indicate complex queries needing smart model
    COMPLEX_QUERY_PATTERNS = [
        # Tool usage indicators
        'github', 'commit', 'summit', 'pr', 'pull request', 'issue', 'repo',
        'implement', 'create', 'build', 'fix', 'debug', 'refactor',
        'code', 'file', 'edit', 'write', 'change', 'modify', 'update',
        # Analysis indicators
        'analyze', 'review', 'explain', 'investigate', 'check',
        'search', 'find', 'look', 'fetch', 'get', 'show',
        # Time-based queries (need GitHub tools)
        'yesterday', 'today', 'last week', 'recent', 'latest', 'what did',
        'what we did', 'what i did', 'status', 'activity',
        # Multi-step indicators
        'then', 'after that', 'also', 'and then', 'next',
    ]

    def __init__(
        self,
        permission_mode: PermissionMode = PermissionMode.BYPASS,
        working_directory: Optional[str] = None,
        timeout: float = 300.0,  # 5 minutes for complex tasks
        project_directories: Optional[list[str]] = None,
        fast_model: str = "haiku",
        smart_model: str = "sonnet",
        use_model_routing: bool = True,
    ):
        """
        Initialize Claude Code interface.

        Args:
            permission_mode: How to handle permission prompts
            working_directory: Working directory for Claude Code
            timeout: Timeout for commands in seconds
            project_directories: List of directories Claude can access/modify
            fast_model: Model for simple queries (haiku)
            smart_model: Model for complex queries (sonnet/opus)
            use_model_routing: Enable automatic model selection
        """
        self.permission_mode = permission_mode
        self.working_directory = working_directory
        self.timeout = timeout
        self.project_directories = project_directories or []
        self.fast_model = fast_model
        self.smart_model = smart_model
        self.use_model_routing = use_model_routing

        # Session management
        self._session_id: Optional[str] = None
        self._conversation_turns = 0

    def _classify_query(self, message: str) -> str:
        """
        Classify query as simple or complex to choose appropriate model.

        Returns:
            'fast' for simple queries, 'smart' for complex ones
        """
        if not self.use_model_routing:
            return 'smart'  # Default to smart model

        message_lower = message.lower()

        # Check for complex patterns
        for pattern in self.COMPLEX_QUERY_PATTERNS:
            if pattern in message_lower:
                return 'smart'

        # Short, simple queries use fast model
        word_count = len(message.split())
        if word_count < 15:
            return 'fast'

        return 'smart'

    def _get_model_for_query(self, message: str) -> str:
        """Get the appropriate model name for the query"""
        query_type = self._classify_query(message)
        return self.fast_model if query_type == 'fast' else self.smart_model

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID"""
        return self._session_id

    @property
    def has_session(self) -> bool:
        """Check if there's an active session"""
        return self._session_id is not None

    def _build_command(
        self,
        resume: bool = True,
        streaming: bool = False,
        additional_args: Optional[list[str]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> list[str]:
        """Build the Claude CLI command"""
        cmd = ['claude', '--print', '--output-format']

        if streaming:
            cmd.extend(['stream-json', '--verbose'])
        else:
            cmd.append('json')

        # Add model selection
        if model:
            cmd.extend(['--model', model])

        # Add permission handling
        if self.permission_mode == PermissionMode.BYPASS:
            cmd.append('--dangerously-skip-permissions')
        elif self.permission_mode != PermissionMode.DEFAULT:
            cmd.extend(['--permission-mode', self.permission_mode.value])

        # Add project directories for code access
        for directory in self.project_directories:
            cmd.extend(['--add-dir', directory])

        # Add system prompt if provided (for new conversations)
        if system_prompt and not resume:
            cmd.extend(['--system-prompt', system_prompt])

        # Resume session if available
        if resume and self._session_id:
            cmd.extend(['--resume', self._session_id])

        # Add any additional arguments
        if additional_args:
            cmd.extend(additional_args)

        return cmd

    async def send(
        self,
        message: str,
        new_conversation: bool = False,
        system_prompt: Optional[str] = None,
    ) -> ClaudeResponse:
        """
        Send a message to Claude Code and get response.

        Args:
            message: The user message to send
            new_conversation: If True, start a new conversation
            system_prompt: Optional system prompt for context

        Returns:
            ClaudeResponse with result
        """
        if new_conversation:
            self._session_id = None
            self._conversation_turns = 0

        # Always select model based on query complexity (even for follow-ups)
        model = self._get_model_for_query(message)

        # Build command with system prompt for new conversations
        cmd = self._build_command(
            resume=not new_conversation,
            system_prompt=system_prompt if new_conversation else None,
            model=model
        )

        start_time = time.time()

        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory
            )

            # Send message and get response
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=message.encode()),
                timeout=self.timeout
            )

            duration_ms = (time.time() - start_time) * 1000

            # Parse response
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return ClaudeResponse(
                    text=f"Error: {error_msg}",
                    session_id=self._session_id or "",
                    is_error=True,
                    duration_ms=duration_ms
                )

            # Parse JSON response
            response_text = stdout.decode().strip()

            # Handle potential multiple JSON objects (streaming format)
            try:
                # Try parsing as single JSON first
                response_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try parsing last line (for stream-json format)
                lines = response_text.strip().split('\n')
                for line in reversed(lines):
                    try:
                        response_data = json.loads(line)
                        if response_data.get('type') == 'result':
                            break
                    except json.JSONDecodeError:
                        continue
                else:
                    return ClaudeResponse(
                        text=response_text,
                        session_id=self._session_id or "",
                        is_error=True,
                        duration_ms=duration_ms
                    )

            # Extract data from response
            self._session_id = response_data.get('session_id', self._session_id)
            self._conversation_turns += 1

            result_text = response_data.get('result', '')

            return ClaudeResponse(
                text=result_text,
                session_id=self._session_id or "",
                is_error=response_data.get('is_error', False),
                duration_ms=duration_ms,
                cost_usd=response_data.get('total_cost_usd', 0),
                model=response_data.get('model', ''),
                num_turns=response_data.get('num_turns', self._conversation_turns),
                raw_response=response_data
            )

        except asyncio.TimeoutError:
            return ClaudeResponse(
                text="Error: Request timed out",
                session_id=self._session_id or "",
                is_error=True,
                duration_ms=self.timeout * 1000
            )

        except Exception as e:
            return ClaudeResponse(
                text=f"Error: {str(e)}",
                session_id=self._session_id or "",
                is_error=True,
                duration_ms=(time.time() - start_time) * 1000
            )

    async def send_streaming(
        self,
        message: str,
        on_chunk: Callable[[str], None],
        new_conversation: bool = False,
    ) -> ClaudeResponse:
        """
        Send message and stream response chunks.

        Args:
            message: The user message
            on_chunk: Callback for each text chunk
            new_conversation: If True, start new conversation

        Returns:
            Final ClaudeResponse
        """
        if new_conversation:
            self._session_id = None
            self._conversation_turns = 0

        cmd = self._build_command(resume=not new_conversation, streaming=True)
        start_time = time.time()
        full_text = ""

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory
            )

            # Send input
            process.stdin.write(message.encode())
            await process.stdin.drain()
            process.stdin.close()

            final_response = {}

            # Read streaming output
            async for line in process.stdout:
                try:
                    chunk_data = json.loads(line.decode())
                    chunk_type = chunk_data.get('type', '')

                    if chunk_type == 'assistant':
                        # Extract text from assistant message
                        content = chunk_data.get('message', {}).get('content', [])
                        for item in content:
                            if item.get('type') == 'text':
                                text = item.get('text', '')
                                full_text += text
                                on_chunk(text)

                    elif chunk_type == 'result':
                        final_response = chunk_data
                        self._session_id = chunk_data.get('session_id', self._session_id)

                except json.JSONDecodeError:
                    continue

            await process.wait()
            duration_ms = (time.time() - start_time) * 1000

            self._conversation_turns += 1

            return ClaudeResponse(
                text=final_response.get('result', full_text),
                session_id=self._session_id or "",
                is_error=final_response.get('is_error', False),
                duration_ms=duration_ms,
                cost_usd=final_response.get('total_cost_usd', 0),
                model=final_response.get('model', ''),
                num_turns=final_response.get('num_turns', self._conversation_turns),
                raw_response=final_response
            )

        except Exception as e:
            return ClaudeResponse(
                text=f"Error: {str(e)}",
                session_id=self._session_id or "",
                is_error=True,
                duration_ms=(time.time() - start_time) * 1000
            )

    async def stream_response(
        self,
        message: str,
        new_conversation: bool = False,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Async generator that yields response chunks.

        Args:
            message: The user message
            new_conversation: If True, start new conversation
            system_prompt: System prompt for new conversations

        Yields:
            StreamChunk objects with text and metadata
        """
        if new_conversation:
            self._session_id = None
            self._conversation_turns = 0

        # Always select model based on query complexity
        model = self._get_model_for_query(message)

        cmd = self._build_command(
            resume=not new_conversation,
            streaming=True,
            system_prompt=system_prompt if new_conversation else None,
            model=model
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.working_directory
        )

        # Send input
        process.stdin.write(message.encode())
        await process.stdin.drain()
        process.stdin.close()

        # Stream output
        async for line in process.stdout:
            try:
                chunk_data = json.loads(line.decode())
                chunk_type = chunk_data.get('type', '')

                if chunk_type == 'assistant':
                    content = chunk_data.get('message', {}).get('content', [])
                    for item in content:
                        if item.get('type') == 'text':
                            yield StreamChunk(
                                text=item.get('text', ''),
                                chunk_type='text',
                                is_final=False
                            )

                elif chunk_type == 'result':
                    self._session_id = chunk_data.get('session_id', self._session_id)
                    self._conversation_turns += 1
                    yield StreamChunk(
                        text=chunk_data.get('result', ''),
                        chunk_type='result',
                        is_final=True
                    )

            except json.JSONDecodeError:
                continue

        await process.wait()

    async def stream_text(
        self,
        message: str,
        new_conversation: bool = False,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Simplified async generator that yields only text strings.
        Ideal for streaming TTS.

        Args:
            message: The user message
            new_conversation: If True, start new conversation
            system_prompt: System prompt for new conversations

        Yields:
            Text strings as they arrive
        """
        async for chunk in self.stream_response(message, new_conversation, system_prompt):
            if chunk.text and chunk.chunk_type == 'text':
                yield chunk.text

    def reset_session(self) -> None:
        """Reset the current session"""
        self._session_id = None
        self._conversation_turns = 0

    def get_session_info(self) -> dict:
        """Get information about current session"""
        return {
            'session_id': self._session_id,
            'has_session': self.has_session,
            'turns': self._conversation_turns,
            'permission_mode': self.permission_mode.value
        }


# Test function
async def test_claude():
    """Test Claude Code integration"""
    print("Testing Claude Code Integration...")

    claude = ClaudeCode(permission_mode=PermissionMode.BYPASS)

    # Test simple query
    print("\n1. Testing simple query...")
    response = await claude.send("What is 2 + 2? Reply with just the number.")
    print(f"   Response: {response.text}")
    print(f"   Session ID: {response.session_id}")
    print(f"   Duration: {response.duration_ms:.0f}ms")

    # Test follow-up (session continuity)
    print("\n2. Testing follow-up (session continuity)...")
    response = await claude.send("What was my previous question? Reply briefly.")
    print(f"   Response: {response.text}")
    print(f"   Same session: {claude.session_id == response.session_id}")

    # Test new conversation
    print("\n3. Testing new conversation...")
    response = await claude.send(
        "Say 'Hello Jarvis user!' and nothing else.",
        new_conversation=True
    )
    print(f"   Response: {response.text}")
    print(f"   New session ID: {response.session_id}")

    # Test streaming
    print("\n4. Testing streaming response...")
    chunks = []

    def on_chunk(text):
        chunks.append(text)
        print(f"   Chunk: {text[:50]}..." if len(text) > 50 else f"   Chunk: {text}")

    response = await claude.send_streaming(
        "Count from 1 to 5, one number per line.",
        on_chunk=on_chunk,
        new_conversation=True
    )
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Final text: {response.text}")

    # Show session info
    print(f"\nSession info: {claude.get_session_info()}")

    print("\nClaude Code integration test complete!")


if __name__ == "__main__":
    asyncio.run(test_claude())
