#!/usr/bin/env python3
"""
Jarvis Voice Agent - Main Entry Point

Phase 1: Core Pipeline
- Press Enter to start listening
- Speak your command
- Claude responds via TTS
- Follow-up within timeout continues conversation

Usage:
    python -m jarvis.main
    # or
    python jarvis/main.py
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.audio.input import AudioInput
from jarvis.audio.output import AudioOutput
from jarvis.audio.vad import VoiceActivityDetector
from jarvis.speech.stt import SpeechToText
from jarvis.speech.corrections import correct_transcription
from jarvis.llm.claude import ClaudeCode, PermissionMode
from jarvis.core.state import StateManager, AgentState, create_phase1_state_machine
from jarvis.core.config import Config, get_config
from jarvis.core.speech_buffer import SpeechBuffer, get_acknowledgment


class JarvisAgent:
    """
    Main Jarvis Voice Agent orchestrator.

    Phase 1: Keyboard-triggered voice interaction
    - Press Enter to start listening
    - Automatic end-of-speech detection
    - Claude Code integration with session persistence
    - TTS response playback
    """

    def __init__(self, config: Config):
        """Initialize the voice agent with configuration"""
        self.config = config

        # Initialize components
        self.audio_input = AudioInput(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            chunk_size=config.audio.chunk_size,
            device_index=config.audio.device_index,
        )

        self.audio_output = AudioOutput(
            voice=config.audio.tts_voice,
            rate=config.audio.tts_rate,
        )

        self.vad = VoiceActivityDetector(
            sample_rate=config.audio.sample_rate,
            energy_threshold=config.audio.silence_threshold,
            max_silence_duration_ms=int(config.audio.silence_duration * 1000),
        )

        self.stt = SpeechToText(
            model_name=config.stt.model_name,
            models_dir=config.stt.models_dir,
            language=config.stt.language,
        )

        self.claude = ClaudeCode(
            permission_mode=PermissionMode.BYPASS,
            timeout=config.claude.timeout,
            working_directory=config.claude.working_directory,
            project_directories=config.claude.project_directories,
            fast_model=config.claude.fast_model,
            smart_model=config.claude.smart_model,
            use_model_routing=config.claude.use_model_routing,
        )

        # State management
        self.state_manager = create_phase1_state_machine()
        self._setup_state_handlers()

        # Control flags
        self._running = False
        self._interrupted = False

    def _setup_state_handlers(self):
        """Set up state change handlers"""
        def on_state_change(old_state: AgentState, new_state: AgentState):
            if self.config.debug:
                print(f"[State] {old_state.name} -> {new_state.name}")

        self.state_manager.on_state_change(on_state_change)

    async def initialize(self) -> bool:
        """Initialize all components with parallel loading where possible"""
        print("Initializing Jarvis...")
        print(f"  Performance mode: streaming TTS, model routing enabled")

        # Test audio input
        try:
            default_device = self.audio_input.get_default_device()
            print(f"  Audio input: {default_device['name']}")
        except Exception as e:
            print(f"  Audio input error: {e}")
            return False

        # Test TTS
        print(f"  TTS voice: {self.config.audio.tts_voice} @ {self.config.audio.tts_rate} wpm")

        # Load STT model (using faster tiny.en by default)
        print(f"  Loading STT model: {self.config.stt.model_name} (optimized for speed)")
        if not self.stt.load_model():
            print("  Failed to load STT model!")
            return False
        print(f"  STT model loaded")

        # Show model routing config
        print(f"  Model routing: {self.config.claude.fast_model} (fast) / {self.config.claude.smart_model} (smart)")

        # Quick Claude connection test (using fast model)
        print("  Warming up Claude connection...")
        response = await self.claude.send("hi", new_conversation=True)
        if response.is_error:
            print(f"  Claude error: {response.text}")
            return False
        print("  Claude Code ready")

        print("\n✓ Jarvis initialized (optimized for speed)")
        return True

    async def listen_and_transcribe(self, prompt: str = "🎤 Listening... (speak now)") -> str:
        """Record audio and transcribe to text"""
        import time
        self.state_manager.transition_to(AgentState.LISTENING)

        # Play listening indicator
        await self.audio_output.play_listening_sound()

        print(prompt)

        # Record until silence - waits for speech to START first
        record_start = time.time()
        audio_data = await self.audio_input.record_until_silence(
            silence_threshold=self.config.audio.silence_threshold,
            silence_duration=self.config.audio.silence_duration,
            max_duration=self.config.audio.max_recording_duration,
            min_speech_duration=0.3,  # At least 0.3s of speech
            wait_for_speech_timeout=15.0,  # Wait up to 15s for user to start speaking
        )
        record_time = time.time() - record_start

        if len(audio_data) < 1000:  # Too short or no speech detected
            print("No speech detected. Please try again.")
            self.state_manager.transition_to(AgentState.IDLE)
            return ""

        # Transcribe
        self.state_manager.transition_to(AgentState.PROCESSING_STT)
        print("Transcribing...")
        transcribe_start = time.time()

        result = await self.stt.transcribe_bytes(
            audio_data,
            sample_rate=self.config.audio.sample_rate
        )
        transcribe_time = time.time() - transcribe_start

        text = result.text.strip()
        if text:
            # Apply transcription corrections for common STT errors
            corrected = correct_transcription(text)

            if self.config.debug:
                print(f"📝 [{transcribe_time:.1f}s] Raw: {text}")
                if corrected != text:
                    print(f"📝 Corrected: {corrected}")
            else:
                print(f"You said: {corrected}")

            return corrected
        else:
            print("Could not transcribe audio")
            return ""

    async def process_with_claude_streaming(self, user_input: str):
        """
        Send user input to Claude and stream response for real-time TTS.

        Yields:
            Text chunks as they arrive from Claude
        """
        self.state_manager.transition_to(AgentState.PROCESSING_LLM)

        # Quick audio feedback that we're processing
        print("⏳ Processing...", end=" ", flush=True)

        # Check if this is a new conversation or follow-up
        is_new = not self.state_manager.context.is_active

        if is_new:
            self.state_manager.context.start()

        # Stream response
        full_text = ""
        first_chunk = True
        async for text in self.claude.stream_text(
            user_input,
            new_conversation=is_new,
            system_prompt=self.config.claude.system_prompt if is_new else None
        ):
            if first_chunk:
                print("✓")  # Show we got response
                first_chunk = False
            full_text += text
            yield text

        # Update context after streaming completes
        self.state_manager.context.session_id = self.claude.session_id
        self.state_manager.context.add_turn(user_input, full_text)

        # Store for question detection
        self._last_response_text = full_text

    async def process_with_claude(self, user_input: str) -> tuple[str, bool]:
        """
        Send user input to Claude and get response (non-streaming fallback).

        Returns:
            tuple of (response_text, is_asking_question)
        """
        self.state_manager.transition_to(AgentState.PROCESSING_LLM)
        print("Thinking...")

        # Check if this is a new conversation or follow-up
        is_new = not self.state_manager.context.is_active

        if is_new:
            self.state_manager.context.start()

        # Send message with system prompt for new conversations
        response = await self.claude.send(
            user_input,
            new_conversation=is_new,
            system_prompt=self.config.claude.system_prompt if is_new else None
        )

        if response.is_error:
            return f"Sorry, I encountered an error: {response.text}", False

        # Update context
        self.state_manager.context.session_id = response.session_id
        self.state_manager.context.add_turn(user_input, response.text)

        return response.text, response.is_asking_question

    async def speak_response(self, text: str) -> None:
        """Speak the response using TTS (non-streaming)"""
        self.state_manager.transition_to(AgentState.SPEAKING)

        if not text:
            return

        print(f"Jarvis: {text}")
        await self.audio_output.speak(text)

        # Play done sound
        await self.audio_output.play_done_sound()

    async def speak_response_streaming(self, text_generator) -> str:
        """
        Speak response in real-time as text streams in.
        Press Enter to skip/interrupt speaking.

        Args:
            text_generator: Async generator yielding text chunks

        Returns:
            Full response text
        """
        self.state_manager.transition_to(AgentState.SPEAKING)

        full_text = ""
        buffer = ""
        sentence_endings = '.!?\n'
        sentences_spoken = 0
        max_sentences = 5  # Limit sentences spoken, then offer more

        print("Jarvis: ", end="", flush=True)

        async for chunk in text_generator:
            full_text += chunk
            buffer += chunk
            print(chunk, end="", flush=True)

            # Speak complete sentences as they arrive (limit to max_sentences)
            while any(end in buffer for end in sentence_endings):
                # Find first sentence ending
                earliest_pos = len(buffer)
                for end in sentence_endings:
                    pos = buffer.find(end)
                    if pos != -1 and pos < earliest_pos:
                        earliest_pos = pos

                # Extract sentence
                sentence = buffer[:earliest_pos + 1].strip()
                buffer = buffer[earliest_pos + 1:].lstrip()

                if sentence and sentences_spoken < max_sentences:
                    # Speak this sentence
                    await self.audio_output.speak(sentence)
                    sentences_spoken += 1

        # Speak any remaining text (if under limit)
        if buffer.strip() and sentences_spoken < max_sentences:
            await self.audio_output.speak(buffer.strip())

        print()  # New line after response

        # If response was long, notify user
        if sentences_spoken >= max_sentences:
            print("(Response truncated for voice. Full text shown above.)")

        await self.audio_output.play_done_sound()

        return full_text

    async def wait_for_followup(self) -> bool:
        """Wait for potential follow-up command"""
        self.state_manager.transition_to(AgentState.WAITING_FOLLOWUP)

        timeout = self.config.follow_up_timeout
        print(f"\n[Press Enter to continue, or wait {timeout:.0f}s to end]")

        try:
            # Wait for Enter key with timeout
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, input),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            print("\nConversation ended. Say 'Hey Jarvis' or press Enter to start a new one.")
            self.state_manager.context.end()
            return False

    def _is_asking_question(self, text: str) -> bool:
        """Check if text is asking a question"""
        if not text:
            return False
        text_lower = text.lower().strip()
        if '?' in text:
            return True
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

    def _extract_first_sentence(self, text: str) -> str:
        """Extract first meaningful sentence from text for fallback TTS"""
        if not text:
            return ""

        # Clean the text first
        cleaned = self.audio_output._clean_text_for_speech(text)
        if not cleaned:
            return ""

        # Find first sentence end
        for i, char in enumerate(cleaned):
            if char in '.!?':
                sentence = cleaned[:i+1].strip()
                if len(sentence) > 10:  # Must be meaningful
                    return sentence

        # No sentence end found, take first N words
        words = cleaned.split()
        if len(words) > 5:
            return ' '.join(words[:20]) + '.'

        return cleaned[:200] if len(cleaned) > 10 else ""

    async def handle_turn(self, is_followup_answer: bool = False) -> bool:
        """
        Handle one conversation turn with immediate acknowledgment and streaming TTS.

        Flow:
        1. Listen and transcribe user speech
        2. Immediately acknowledge ("Checking GitHub.", "On it.", etc.)
        3. Stream response from Claude
        4. Speak text as it arrives using smart buffering
        5. Guarantee voice output even if streaming fails

        Args:
            is_followup_answer: If True, this is an answer to Jarvis's question

        Returns:
            True if conversation should continue
        """
        try:
            # 1. Listen and transcribe
            if is_followup_answer:
                prompt = "🎤 Your answer..."
            else:
                prompt = "🎤 Listening..."

            user_input = await self.listen_and_transcribe(prompt=prompt)
            if not user_input:
                if is_followup_answer:
                    print("No response. Ending.")
                    self.state_manager.context.end()
                self.state_manager.transition_to(AgentState.IDLE)
                return False

            # 2. IMMEDIATE acknowledgment - user knows they were heard
            self.state_manager.transition_to(AgentState.PROCESSING_LLM)
            ack = get_acknowledgment(user_input)
            print(f"🤖 {ack}")

            # Speak acknowledgment (fast, non-blocking feel)
            await self.audio_output.speak(ack, max_length=50)

            # 3. Check conversation state
            is_new = not self.state_manager.context.is_active
            if is_new:
                self.state_manager.context.start()

            # 4. Stream response and speak with smart buffering
            speech_buffer = SpeechBuffer(min_words=6, max_words=25, timeout_seconds=2.0)
            full_text = ""
            first_chunk = True
            max_spoken = 5  # Maximum phrases to speak

            print("Jarvis: ", end="", flush=True)

            async for chunk in self.claude.stream_text(
                user_input,
                new_conversation=is_new,
                system_prompt=self.config.claude.system_prompt if is_new else None
            ):
                if first_chunk:
                    self.state_manager.transition_to(AgentState.SPEAKING)
                    first_chunk = False

                full_text += chunk
                print(chunk, end="", flush=True)

                # Add to smart buffer
                speech_buffer.add(chunk)

                # Check if we have speakable content
                if speech_buffer._total_spoken < max_spoken:
                    speakable = speech_buffer.get_speakable()
                    if speakable:
                        await self.audio_output.speak(speakable, max_length=150)

            print()  # Newline after streaming

            # 5. Flush remaining buffer
            if speech_buffer._total_spoken < max_spoken:
                remaining = speech_buffer.flush()
                if remaining:
                    await self.audio_output.speak(remaining, max_length=150)

            # 6. GUARANTEE voice output - if nothing was spoken, speak summary
            if not speech_buffer.has_spoken() and full_text:
                # Extract first meaningful sentence as fallback
                fallback = self._extract_first_sentence(full_text)
                if fallback:
                    print(f"(Speaking summary)")
                    await self.audio_output.speak(fallback, max_length=200)

            # Update context
            self.state_manager.context.session_id = self.claude.session_id
            self.state_manager.context.add_turn(user_input, full_text)

            await self.audio_output.play_done_sound()

            # Check if response asks a question
            is_asking_question = self._is_asking_question(full_text)

            # 7. Handle follow-up if Jarvis asked a question
            if is_asking_question and self.config.auto_continue_on_question:
                print("\n💬 Respond to Jarvis...")
                await asyncio.sleep(0.5)
                return await self.handle_turn(is_followup_answer=True)

            # 8. Wait for follow-up
            return await self.wait_for_followup()

        except Exception as e:
            print(f"Error: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            self.state_manager.transition_to(AgentState.ERROR, force=True)
            await asyncio.sleep(1)
            self.state_manager.transition_to(AgentState.IDLE, force=True)
            return False

    async def run(self) -> None:
        """Main agent loop"""
        self._running = True

        print("\n" + "=" * 50)
        print("   JARVIS - Smart Voice Assistant")
        print("=" * 50)
        print("\nCapabilities:")
        print("  - GitHub: commits, PRs, issues")
        print("  - Code: read, edit, create files")
        print("  - Run commands: build, test, git")
        print("\nProject directories:")
        for d in self.config.claude.project_directories:
            print(f"  - {d}")
        print("\nPress Enter to start speaking, Ctrl+C to exit")
        print("-" * 50)

        while self._running:
            try:
                self.state_manager.transition_to(AgentState.IDLE)

                # Wait for Enter key to start
                print("\n[Press Enter to speak...]")
                await asyncio.get_event_loop().run_in_executor(None, input)

                if not self._running:
                    break

                # Handle conversation turns
                continue_conversation = True
                while continue_conversation and self._running:
                    continue_conversation = await self.handle_turn(is_followup_answer=False)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"\nError: {e}")
                if self.config.debug:
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(1)

        await self.shutdown()

    async def shutdown(self) -> None:
        """Clean up resources"""
        print("\nShutting down Jarvis...")
        self._running = False

        # Interrupt any ongoing speech
        await self.audio_output.interrupt()

        # Clean up audio
        self.audio_input.cleanup()

        print("Goodbye!")

    def stop(self) -> None:
        """Stop the agent"""
        self._running = False


async def main():
    """Main entry point"""
    # Load configuration
    config = get_config()

    # Enable debug mode via command line
    if "--debug" in sys.argv:
        config.debug = True

    # Create and initialize agent
    agent = JarvisAgent(config)

    # Set up signal handlers
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        agent.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize
    if not await agent.initialize():
        print("Failed to initialize Jarvis")
        sys.exit(1)

    # Run
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
