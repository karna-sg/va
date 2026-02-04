"""
Jarvis Orchestrator - Main Voice Agent Loop

Coordinates all components:
- Voice I/O (audio input, STT, TTS)
- 3-Tier Intent Router (FAISS -> Local LLM -> Claude CLI)
- Conversation management (context, entities, history)
- LLM backend (Claude Code CLI)
- MCP tool management
- Short-term memory
- Session persistence (SQLite)
- State machine
"""

import asyncio
import os
import time
from pathlib import Path

from jarvis.audio.input import AudioInput
from jarvis.audio.output import AudioOutput
from jarvis.audio.vad import VoiceActivityDetector
from jarvis.speech.stt import SpeechToText
from jarvis.speech.stt_deepgram import DeepgramSTT, DeepgramSTTFallback
from jarvis.speech.corrections import correct_transcription
from jarvis.speech.tts_elevenlabs import ElevenLabsTTS, ElevenLabsTTSFallback
from jarvis.llm.claude import ClaudeCode, PermissionMode
from jarvis.core.state import StateManager, AgentState, create_phase1_state_machine
from jarvis.core.config import Config
from jarvis.core.speech_buffer import SpeechBuffer, get_acknowledgment
from jarvis.core.conversation import ConversationManager
from jarvis.tools.mcp_manager import MCPClientManager
from jarvis.tools.direct_executor import DirectExecutor, YES_WORDS, NO_WORDS
from jarvis.memory.short_term import ShortTermMemory
from jarvis.memory.session_store import SessionStore
from jarvis.memory.long_term import LongTermMemory
from jarvis.memory.temporal import TemporalMemory
from jarvis.workflows.planner import WorkflowPlanner
from jarvis.training.collect import TrainingDataCollector

# Tier 1 routing (optional - gracefully degrades if deps missing)
try:
    from jarvis.intents.fast_router import FastRouter
    TIER1_AVAILABLE = True
except ImportError:
    TIER1_AVAILABLE = False

# Tier 2 local LLM routing (optional - gracefully degrades)
try:
    from jarvis.llm.local_router import LocalRouter
    TIER2_AVAILABLE = True
except ImportError:
    TIER2_AVAILABLE = False


class Orchestrator:
    """
    Main Jarvis Voice Agent orchestrator.

    Manages the full voice interaction pipeline:
    1. Listen (STT) -> 2. Route -> 3. Execute -> 4. Speak (TTS)

    Phase 1: Keyboard-triggered, Claude CLI backend
    Phase 2+: Wake word, Tier 1/2 local routing, direct MCP calls
    """

    def __init__(self, config: Config):
        self.config = config

        # --- Voice I/O ---
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

        # TTS: ElevenLabs or macOS fallback
        self.tts = None
        if config.audio.use_elevenlabs:
            try:
                self.tts = ElevenLabsTTS(
                    voice_id=config.audio.elevenlabs_voice,
                    model_id=config.audio.elevenlabs_model,
                )
                print("  TTS: ElevenLabs (%s)" % config.audio.elevenlabs_voice)
            except Exception as e:
                print("  ElevenLabs unavailable: %s" % e)

        if self.tts is None:
            self.tts = ElevenLabsTTSFallback(
                voice=config.audio.tts_voice,
                rate=config.audio.tts_rate,
            )
            print("  TTS: macOS (%s)" % config.audio.tts_voice)

        self.vad = VoiceActivityDetector(
            sample_rate=config.audio.sample_rate,
            energy_threshold=config.audio.silence_threshold,
            max_silence_duration_ms=int(config.audio.silence_duration * 1000),
        )

        # --- STT: Deepgram streaming or Whisper fallback ---
        self.deepgram_stt = None
        self.use_deepgram = False
        if config.stt.use_deepgram:
            if os.getenv("DEEPGRAM_API_KEY"):
                try:
                    self.deepgram_stt = DeepgramSTT(
                        model=config.stt.deepgram_model,
                        language=config.stt.deepgram_language,
                        endpointing=config.stt.deepgram_endpointing,
                    )
                    self.use_deepgram = True
                    print("  STT: Deepgram streaming (%s)" % config.stt.deepgram_model)
                except Exception as e:
                    print("  Deepgram STT unavailable: %s" % e)

        self.stt = SpeechToText(
            model_name=config.stt.model_name,
            models_dir=config.stt.models_dir,
            language=config.stt.language,
        )
        if not self.use_deepgram:
            print("  STT: Whisper local (%s)" % config.stt.model_name)

        # --- LLM Backend ---
        self.claude = ClaudeCode(
            permission_mode=PermissionMode.BYPASS,
            timeout=config.claude.timeout,
            working_directory=config.claude.working_directory,
            project_directories=config.claude.project_directories,
            fast_model=config.claude.fast_model,
            smart_model=config.claude.smart_model,
            use_model_routing=config.claude.use_model_routing,
        )
        print("  LLM: Claude Code CLI")

        # --- MCP Tools ---
        mcp_config_path = str(Path(config.claude.working_directory) / '.mcp.json')
        self.mcp = MCPClientManager(config_path=mcp_config_path)

        # --- Tier 1 Intent Router (FAISS) ---
        self.fast_router = None
        self.direct_executor = None
        if TIER1_AVAILABLE:
            self.fast_router = FastRouter()
            self.direct_executor = DirectExecutor(
                claude=self.claude,
                defaults={
                    'default_repo': config.claude.default_repos[0] if config.claude.default_repos else 'curiescious',
                    'default_owner': config.claude.github_owner,
                    'default_channel': '#general',
                },
            )

        # --- Tier 2 Local LLM Router (MLX) ---
        self.local_router = None
        if TIER2_AVAILABLE:
            self.local_router = LocalRouter()

        # --- Workflow Engine ---
        self.workflow_planner = WorkflowPlanner(
            claude=self.claude,
            config=config,
            on_progress=self._on_workflow_progress,
        )

        # --- Conversation & Memory ---
        self.conversation = ConversationManager(
            max_turns=config.max_conversation_turns
        )
        self.memory = ShortTermMemory()
        self.session_store = SessionStore()
        self.long_term_memory = LongTermMemory()
        self.temporal_memory = TemporalMemory()

        # --- Training Data ---
        self.training_collector = TrainingDataCollector()

        # --- State Machine ---
        self.state_manager = create_phase1_state_machine()
        self._setup_state_handlers()

        # --- Control ---
        self._running = False
        self._interrupted = False

    def _setup_state_handlers(self):
        """Set up state change handlers"""
        def on_state_change(old_state: AgentState, new_state: AgentState):
            if self.config.debug:
                print("[State] %s -> %s" % (old_state.name, new_state.name))

        self.state_manager.on_state_change(on_state_change)

    async def initialize(self) -> bool:
        """Initialize all components"""
        print("Initializing Kat...")
        print("  Performance mode: streaming TTS, model routing enabled")

        # Audio input
        try:
            default_device = self.audio_input.get_default_device()
            print("  Audio input: %s" % default_device['name'])
        except Exception as e:
            print("  Audio input error: %s" % e)
            return False

        # TTS
        print("  TTS voice: %s @ %s wpm" % (self.config.audio.tts_voice, self.config.audio.tts_rate))

        # STT model + pre-connect Deepgram WebSocket
        if self.use_deepgram:
            print("  STT: Deepgram streaming (pre-connecting...)")
            await self.deepgram_stt.connect()
        else:
            print("  Loading Whisper STT model: %s" % self.config.stt.model_name)
            if not self.stt.load_model():
                print("  Failed to load STT model!")
                return False
            print("  STT model loaded")

        # Model routing
        print("  Model routing: %s (fast) / %s (smart)" % (
            self.config.claude.fast_model, self.config.claude.smart_model))

        # MCP servers
        await self.mcp.initialize()

        # Session store (SQLite)
        await self.session_store.initialize()

        # Long-term memory (FAISS-indexed preferences + patterns)
        if self.long_term_memory.initialize():
            stats = self.long_term_memory.get_stats()
            print("  Long-term memory: %d records (%d preferences)" % (
                stats['total'], stats['preferences']))
        else:
            print("  Long-term memory: unavailable")

        # Temporal memory (time-bound facts)
        if self.temporal_memory.initialize():
            stats = self.temporal_memory.get_stats()
            print("  Temporal memory: %d active facts, %d patterns" % (
                stats['facts_active'], stats['patterns_total']))
        else:
            print("  Temporal memory: unavailable")

        # Training data collector
        if self.training_collector.initialize():
            stats = self.training_collector.get_stats()
            print("  Training data: %d samples, %d logs" % (
                stats['training_samples']['total'],
                stats['conversation_logs']))
        else:
            print("  Training collector: unavailable")

        # Tier 1 intent router (FAISS)
        if self.fast_router:
            print("  Loading Tier 1 intent router...")
            if await self.fast_router.initialize():
                info = self.fast_router.get_catalog_info()
                print("  Tier 1 ready: %d intents, %d phrases" % (
                    info['num_intents'], info['total_phrases']))
            else:
                print("  Tier 1 unavailable (will use Claude for all queries)")
                self.fast_router = None
        else:
            print("  Tier 1 not available (install sentence-transformers faiss-cpu pyyaml)")

        # Tier 2 local LLM router (MLX)
        if self.local_router:
            print("  Loading Tier 2 local LLM...")
            if self.local_router.initialize():
                stats = self.local_router.get_stats()
                print("  Tier 2 ready: %s" % stats['model'])
            else:
                print("  Tier 2 unavailable (will fall through to Claude)")
                self.local_router = None
        else:
            print("  Tier 2 not available (install mlx mlx-lm)")

        # Workflow engine
        workflows = self.workflow_planner.get_available_workflows()
        print("  Workflows: %d templates (%s)" % (
            len(workflows),
            ", ".join(w['name'] for w in workflows),
        ))

        # Claude warmup
        print("  Warming up Claude Code CLI...")
        response = await self.claude.send("hi", new_conversation=True)
        if response.is_error:
            print("  Claude error: %s" % response.text)
            return False
        print("  Claude Code ready (%0.fms)" % response.duration_ms)

        print("\nKat initialized (optimized for speed)")
        return True

    # -------------------------------------------------------------------------
    # Voice I/O
    # -------------------------------------------------------------------------

    async def listen_and_transcribe(self, prompt: str = "Listening... (speak now)") -> str:
        """Record audio and transcribe to text"""
        self.state_manager.transition_to(AgentState.LISTENING)
        await self.audio_output.play_listening_sound()
        print(prompt)

        if self.use_deepgram and self.deepgram_stt:
            return await self._listen_with_deepgram()

        return await self._listen_with_whisper()

    async def _listen_with_deepgram(self) -> str:
        """Listen using Deepgram streaming STT (<300ms latency)"""
        import time
        start_time = time.time()

        final_text = ""
        interim_text = ""

        try:
            async for result in self.deepgram_stt.transcribe_stream(
                timeout=self.config.audio.max_recording_duration
            ):
                current = result.text.strip()
                if current and current != interim_text:
                    interim_text = current
                    prefix = ">" if result.is_final else "..."
                    print("\r%s %s          " % (prefix, interim_text), end="", flush=True)

                if result.is_final:
                    final_text = current

                if result.speech_final:
                    final_text = current
                    break

        except Exception as e:
            print("\nDeepgram error: %s, falling back to Whisper" % e)
            return await self._listen_with_whisper()

        elapsed = time.time() - start_time

        if final_text:
            corrected = correct_transcription(final_text)
            if self.config.debug:
                print("\r[%0.1fs] Raw: %s" % (elapsed, final_text))
                if corrected != final_text:
                    print("Corrected: %s" % corrected)
            else:
                print("\rYou said: %s          " % corrected)
            return corrected
        else:
            print("\rNo speech detected. Please try again.")
            self.state_manager.transition_to(AgentState.IDLE)
            return ""

    async def _listen_with_whisper(self) -> str:
        """Listen using Whisper local STT (fallback)"""
        import time

        record_start = time.time()
        audio_data = await self.audio_input.record_until_silence(
            silence_threshold=self.config.audio.silence_threshold,
            silence_duration=self.config.audio.silence_duration,
            max_duration=self.config.audio.max_recording_duration,
            min_speech_duration=0.3,
            wait_for_speech_timeout=15.0,
        )
        record_time = time.time() - record_start

        if len(audio_data) < 1000:
            print("No speech detected. Please try again.")
            self.state_manager.transition_to(AgentState.IDLE)
            return ""

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
            corrected = correct_transcription(text)
            if self.config.debug:
                print("[%0.1fs] Raw: %s" % (transcribe_time, text))
                if corrected != text:
                    print("Corrected: %s" % corrected)
            else:
                print("You said: %s" % corrected)
            return corrected
        else:
            print("Could not transcribe audio")
            return ""

    # -------------------------------------------------------------------------
    # Response Processing
    # -------------------------------------------------------------------------

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

        buffer = SpeechBuffer()
        cleaned = buffer._clean_for_speech(text)
        if not cleaned:
            return ""

        for i, char in enumerate(cleaned):
            if char in '.!?':
                sentence = cleaned[:i + 1].strip()
                if len(sentence) > 10:
                    return sentence

        words = cleaned.split()
        if len(words) > 5:
            return ' '.join(words[:20]) + '.'

        return cleaned[:200] if len(cleaned) > 10 else ""

    # -------------------------------------------------------------------------
    # Turn Handling
    # -------------------------------------------------------------------------

    async def handle_turn(self, is_followup_answer: bool = False) -> bool:
        """
        Handle one conversation turn with 3-tier routing.

        Flow:
        1. Listen and transcribe
        2. Tier 1 routing (FAISS, <1ms) - try fast match first
        3. If Tier 1 matched:
           a. Execute directly (local response or targeted Claude call)
           b. Speak result
        4. If Tier 1 missed or needs Claude:
           a. Immediate acknowledgment
           b. Stream full response from Claude CLI
           c. Speak with smart buffering
        5. Update conversation + memory
        6. Handle follow-ups

        Returns:
            True if conversation should continue
        """
        try:
            # 1. Listen and transcribe
            prompt = "Your answer..." if is_followup_answer else "Listening..."
            user_input = await self.listen_and_transcribe(prompt=prompt)
            if not user_input:
                if is_followup_answer:
                    print("No response. Ending.")
                    self.conversation.end()
                self.state_manager.transition_to(AgentState.IDLE)
                return False

            # 2. Tier 1 routing (FAISS embedding search)
            routing = None
            if self.fast_router and self.fast_router.is_ready and not is_followup_answer:
                self.state_manager.transition_to(AgentState.ROUTING)
                route_start = time.time()
                routing = self.fast_router.route(user_input)
                route_ms = (time.time() - route_start) * 1000

                if routing:
                    if self.config.debug:
                        print("[Tier 1] %s (%.3f) in %.1fms -> %s" % (
                            routing.intent, routing.confidence,
                            route_ms, routing.matched_phrase))
                    else:
                        print("[T1: %s %.0f%%]" % (routing.intent, routing.confidence * 100))

            # 2b. Tier 2 routing (local LLM) - when Tier 1 is uncertain
            if not routing and self.local_router and self.local_router.is_ready and not is_followup_answer:
                # Already in ROUTING state from Tier 1 attempt (or transition now)
                route_start = time.time()
                tier2_result = self.local_router.route(
                    user_input,
                    context=self._build_memory_context(user_input),
                )
                route_ms = (time.time() - route_start) * 1000

                if tier2_result and tier2_result.intent != 'unknown' and tier2_result.confidence >= 0.7:
                    # Check for missing slots - ask user before executing
                    if tier2_result.missing_slots:
                        return await self._handle_slot_filling(
                            tier2_result, user_input)

                    # Convert to RoutingResult
                    routing = self.local_router.to_routing_result(tier2_result, user_input)

                    if self.config.debug:
                        print("[Tier 2] %s (%.3f) in %.1fms - %s" % (
                            routing.intent, routing.confidence,
                            route_ms, tier2_result.reasoning))
                    else:
                        print("[T2: %s %.0f%%]" % (routing.intent, routing.confidence * 100))

            # 3. Confirmation gate - ask before executing non-local intents
            if routing and self.direct_executor:
                confirm_prompt = self.direct_executor.get_confirmation_prompt(routing)
                if confirm_prompt:
                    confirmed = await self._confirm_action(confirm_prompt)
                    if not confirmed:
                        self.state_manager.transition_to(AgentState.IDLE)
                        return await self.wait_for_followup()

            # 3a. Workflow intents - execute via workflow engine
            if routing and routing.intent.startswith('workflow.'):
                return await self._handle_workflow(routing, user_input)

            # 3b. Tier 1 or Tier 2 matched - try direct execution
            if routing and self.direct_executor and not routing.needs_claude:
                return await self._handle_direct_execution(routing, user_input)

            # 4. Fall through to Claude CLI (Tier 3)
            # Use routing info for smarter acknowledgment if available
            if routing:
                ack = routing.response_template or get_acknowledgment(user_input)
            else:
                ack = get_acknowledgment(user_input)

            self.state_manager.transition_to(AgentState.PROCESSING_LLM)
            print(">> %s" % ack)
            await self.tts.speak(ack)

            # Conversation state
            is_new = not self.conversation.is_active
            if is_new:
                self.conversation.start()

            # Stream response + speak with smart buffering
            full_text = await self._stream_claude_response(user_input, is_new)

            # Update conversation + memory
            self.conversation.session_id = self.claude.session_id
            self.conversation.add_turn(user_input, full_text)
            self.memory.store('last_query', user_input, category='fact', ttl=3600)
            self.memory.store('last_response', full_text[:500], category='fact', ttl=3600)

            # Log for training + temporal patterns
            self._log_turn(user_input, full_text, routing, tier=3)

            # Done sound
            await self.audio_output.play_done_sound()

            # Auto-continue if Kat asked a question
            if self._is_asking_question(full_text) and self.config.auto_continue_on_question:
                print("\nRespond to Kat...")
                await asyncio.sleep(0.5)
                return await self.handle_turn(is_followup_answer=True)

            # Wait for follow-up
            return await self.wait_for_followup()

        except Exception as e:
            print("Error: %s" % e)
            if self.config.debug:
                import traceback
                traceback.print_exc()
            self.state_manager.transition_to(AgentState.ERROR, force=True)
            await asyncio.sleep(1)
            self.state_manager.transition_to(AgentState.IDLE, force=True)
            return False

    async def _confirm_action(self, prompt: str) -> bool:
        """
        Ask the user to confirm an action before executing.

        Speaks the confirmation prompt, listens for yes/no.
        Returns True if confirmed, False if cancelled.
        """
        self.state_manager.transition_to(AgentState.CONFIRMING)

        # Speak the confirmation prompt (short text, use fast TTS)
        print("Kat: %s" % prompt)
        await self.tts.speak_short(prompt)

        # Listen for yes/no
        answer = await self.listen_and_transcribe(prompt="Yes or no?")
        if not answer:
            print("No response. Cancelling.")
            return False

        # Parse the answer
        answer_lower = answer.lower().strip().rstrip('.!?')

        # Check for explicit yes/no
        for word in YES_WORDS:
            if word in answer_lower:
                return True
        for word in NO_WORDS:
            if word in answer_lower:
                print("Cancelled.")
                await self.tts.speak_short("Cancelled.")
                return False

        # Ambiguous answer - default to yes if short, cancel if long/unclear
        if len(answer_lower.split()) <= 2:
            # Short answer, probably affirmative
            return True

        print("Didn't catch that. Cancelling to be safe.")
        await self.tts.speak_short("Didn't catch that. Cancelling.")
        return False

    async def _handle_direct_execution(self, routing, user_input: str) -> bool:
        """
        Handle a Tier 1 matched intent via direct execution.

        Skips full Claude conversation flow for instant responses.
        """
        exec_start = time.time()

        result = await self.direct_executor.execute(routing, user_input)

        if result is None:
            # Direct executor can't handle it, fall through to Claude
            if self.config.debug:
                print("[Tier 1] Direct executor declined, falling through to Claude")
            self.state_manager.transition_to(AgentState.PROCESSING_LLM)
            ack = get_acknowledgment(user_input)
            print(">> %s" % ack)
            await self.tts.speak(ack)

            is_new = not self.conversation.is_active
            if is_new:
                self.conversation.start()

            full_text = await self._stream_claude_response(user_input, is_new)

            self.conversation.session_id = self.claude.session_id
            self.conversation.add_turn(user_input, full_text)
            self.memory.store('last_query', user_input, category='fact', ttl=3600)
            self.memory.store('last_response', full_text[:500], category='fact', ttl=3600)
            await self.audio_output.play_done_sound()
            return await self.wait_for_followup()

        exec_ms = (time.time() - exec_start) * 1000

        # Direct execution succeeded
        self.state_manager.transition_to(AgentState.SPEAKING)
        print("Kat: %s" % result.text)
        print("[%s, %.0fms]" % (result.source, exec_ms))
        await self.tts.speak(result.spoken_text)

        # Update conversation + memory
        if not self.conversation.is_active:
            self.conversation.start()
        self.conversation.add_turn(user_input, result.text)
        self.memory.store('last_query', user_input, category='fact', ttl=3600)
        self.memory.store('last_response', result.text[:500], category='fact', ttl=3600)

        # Log successful Tier 1 routing for training
        self._log_turn(user_input, result.text, routing, tier=1)
        self.training_collector.record_confirmed_routing(
            utterance=user_input,
            intent=routing.intent,
            params=routing.params,
            confidence=routing.confidence,
        )

        await self.audio_output.play_done_sound()
        return await self.wait_for_followup()

    async def _handle_slot_filling(self, tier2_result, user_input: str) -> bool:
        """
        Handle missing slots by asking the user via voice.

        When Tier 2 identifies missing parameters (e.g. "post to slack" but
        which channel?), ask the user and then re-route with filled slots.
        """
        missing = tier2_result.missing_slots
        if not missing:
            return False

        # Ask for the first missing slot
        slot_name = missing[0]
        question = "Which %s?" % slot_name.replace('_', ' ')

        self.state_manager.transition_to(AgentState.SPEAKING)
        print("Kat: %s" % question)
        await self.tts.speak(question)

        # Listen for the answer
        answer = await self.listen_and_transcribe(prompt="Your answer...")
        if not answer:
            print("No answer received.")
            self.state_manager.transition_to(AgentState.IDLE)
            return False

        # Fill the slot
        tier2_result.params[slot_name] = answer.strip()
        tier2_result.missing_slots.remove(slot_name)

        # If more slots missing, recurse
        if tier2_result.missing_slots:
            return await self._handle_slot_filling(tier2_result, user_input)

        # All slots filled - convert to RoutingResult and execute
        routing = self.local_router.to_routing_result(tier2_result, user_input)

        if routing.intent.startswith('workflow.'):
            return await self._handle_workflow(routing, user_input)
        elif not routing.needs_claude and self.direct_executor:
            return await self._handle_direct_execution(routing, user_input)
        else:
            # Fall through to Claude with filled params
            enriched_input = "%s (params: %s)" % (user_input, tier2_result.params)
            self.state_manager.transition_to(AgentState.PROCESSING_LLM)
            ack = get_acknowledgment(user_input)
            print(">> %s" % ack)
            await self.tts.speak(ack)

            is_new = not self.conversation.is_active
            if is_new:
                self.conversation.start()

            full_text = await self._stream_claude_response(enriched_input, is_new)
            self.conversation.session_id = self.claude.session_id
            self.conversation.add_turn(user_input, full_text)
            await self.audio_output.play_done_sound()
            return await self.wait_for_followup()

    async def _handle_workflow(self, routing, user_input: str) -> bool:
        """
        Handle a workflow.* intent via the DAG workflow engine.

        Runs multi-step workflows with parallel execution and progress updates.
        """
        from jarvis.workflows.templates import get_template

        self.state_manager.transition_to(AgentState.EXECUTING_WORKFLOW)

        # Extract the workflow action (e.g. 'daily_status' from 'workflow.daily_status')
        action = routing.intent.split('.', 1)[1] if '.' in routing.intent else routing.intent
        template = get_template(action)

        if template:
            # Speak the start message
            spoken_start = template.get('spoken_start', 'Working on it.')
            print(">> %s" % spoken_start)
            await self.tts.speak(spoken_start)

            # Execute the pre-defined workflow
            result = await self.workflow_planner.execute_workflow(
                action,
                params=routing.params,
                context={'memory': self.memory},
            )
        else:
            # Dynamic workflow - plan and execute
            print(">> Planning workflow...")
            await self.tts.speak("Let me figure out the steps.")

            result = await self.workflow_planner.plan_and_execute(
                user_input,
                context={'memory': self.memory},
            )

        # Report result
        self.state_manager.transition_to(AgentState.SPEAKING)
        print("[Workflow] %s" % result.summary)

        if result.success:
            spoken_done = ""
            if template:
                spoken_done = template.get('spoken_done', 'Done.')
            else:
                spoken_done = "Done. All steps completed."

            # Include final result in response if available
            response_text = result.final_result or spoken_done
            if isinstance(response_text, str) and len(response_text) > 200:
                # Summarize long results for voice
                print("Kat: %s" % response_text)
                await self.tts.speak(spoken_done)
            else:
                print("Kat: %s" % (response_text or spoken_done))
                await self.tts.speak(str(response_text or spoken_done))
        else:
            error_msg = "Workflow failed: %s" % (result.error or "unknown error")
            if result.failed_steps:
                failed_names = [s.name for s in result.failed_steps]
                error_msg = "Failed at: %s" % ", ".join(failed_names)
            print("Kat: %s" % error_msg)
            await self.tts.speak("Sorry, the workflow had an issue. %s" % error_msg)

        # Update conversation + memory
        if not self.conversation.is_active:
            self.conversation.start()
        self.conversation.add_turn(user_input, result.summary)
        self.memory.store('last_query', user_input, category='fact', ttl=3600)
        self.memory.store('last_response', result.summary, category='fact', ttl=3600)
        self.memory.store_tool_result(
            'workflow_%s' % routing.intent,
            result.final_result,
            summary=result.summary,
        )

        # Persist workflow result
        self.session_store.store_workflow_result(
            workflow_name=routing.intent,
            result={
                'success': result.success,
                'summary': result.summary,
                'step_results': {k: str(v)[:500] for k, v in result.step_results.items()},
            },
        )

        await self.audio_output.play_done_sound()
        return await self.wait_for_followup()

    def _on_workflow_progress(self, step_name: str, message: str) -> None:
        """Callback for real-time workflow progress updates"""
        print("  [%s] %s" % (step_name, message))

    def _log_turn(self, user_input: str, response: str,
                  routing=None, tier: int = 3) -> None:
        """Log a conversation turn for training and temporal pattern tracking"""
        # Training data
        self.training_collector.log_conversation(
            user_input=user_input,
            response=response,
            session_id=self.claude.session_id if hasattr(self.claude, 'session_id') else None,
            intent=routing.intent if routing else None,
            tier=tier,
            confidence=routing.confidence if routing else None,
        )

        # Temporal pattern tracking
        intent = routing.intent if routing else 'unknown'
        self.temporal_memory.record_command(user_input[:100], intent=intent)

        # Store current task context as temporal fact
        self.temporal_memory.store_fact(
            key='last_command',
            content=user_input[:200],
            fact_type='session',
        )

    def _build_memory_context(self, query: str = "") -> str:
        """
        Build a memory context string to inject into LLM calls.

        Combines context from all memory tiers:
        1. Short-term (in-memory): recent entities, tool results
        2. Temporal: active session/daily facts, time patterns
        3. Long-term: preferences, relevant knowledge
        4. Conversation: recent turns, active entities
        """
        parts = []

        # Conversation context
        conv_ctx = self.conversation.get_context_summary()
        if conv_ctx:
            parts.append(conv_ctx)

        # Short-term memory
        stm_ctx = self.memory.get_context_for_llm()
        if stm_ctx:
            parts.append(stm_ctx)

        # Temporal facts
        temporal_ctx = self.temporal_memory.get_context_for_llm()
        if temporal_ctx:
            parts.append(temporal_ctx)

        # Long-term memory (semantic recall based on query)
        if self.long_term_memory.is_ready and query:
            ltm_ctx = self.long_term_memory.get_context_for_llm(query)
            if ltm_ctx:
                parts.append(ltm_ctx)

        return "\n".join(parts) if parts else ""

    async def _stream_claude_response(self, user_input: str, is_new: bool) -> str:
        """Stream response from Claude CLI with smart buffering and TTS."""
        speech_buffer = SpeechBuffer(min_words=4, max_words=18, timeout_seconds=1.5)
        full_text = ""
        first_chunk = True
        max_spoken = 5

        # Build system prompt with memory context injection
        system_prompt = None
        if is_new:
            memory_ctx = self._build_memory_context(user_input)
            if memory_ctx:
                system_prompt = "%s\n\n## Context\n%s" % (
                    self.config.claude.system_prompt, memory_ctx)
            else:
                system_prompt = self.config.claude.system_prompt

        print("Kat: ", end="", flush=True)

        async for chunk in self.claude.stream_text(
            user_input,
            new_conversation=is_new,
            system_prompt=system_prompt
        ):
            if first_chunk:
                self.state_manager.transition_to(AgentState.SPEAKING)
                first_chunk = False

            full_text += chunk
            print(chunk, end="", flush=True)

            speech_buffer.add(chunk)

            if speech_buffer._total_spoken < max_spoken:
                speakable = speech_buffer.get_speakable()
                if speakable:
                    await self.tts.speak(speakable)

        print()

        # Flush remaining buffer
        if speech_buffer._total_spoken < max_spoken:
            remaining = speech_buffer.flush()
            if remaining:
                await self.tts.speak(remaining)

        # Guarantee voice output
        if not speech_buffer.has_spoken() and full_text:
            fallback = self._extract_first_sentence(full_text)
            if fallback:
                print("(Speaking summary)")
                await self.tts.speak(fallback)

        return full_text

    async def wait_for_followup(self) -> bool:
        """Wait for potential follow-up command"""
        self.state_manager.transition_to(AgentState.WAITING_FOLLOWUP)

        timeout = self.config.follow_up_timeout
        print("\n[Press Enter to continue, or wait %0.fs to end]" % timeout)

        try:
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, input),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            print("\nConversation ended.")
            self.conversation.end()
            return False

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    async def run(self) -> None:
        """Main agent loop"""
        self._running = True

        print("\n" + "=" * 50)
        print("   KAT - Smart Voice Assistant")
        print("=" * 50)
        print("\nCapabilities:")
        print("  - GitHub: commits, PRs, issues")
        print("  - Code: read, edit, create files")
        print("  - Run commands: build, test, git")
        print("  - Workflows: daily status, PR review, sprint planning")

        # Show routing info
        tiers = []
        if self.fast_router and self.fast_router.is_ready:
            info = self.fast_router.get_catalog_info()
            tiers.append("Tier 1 (FAISS, %d intents)" % info['num_intents'])
        if self.local_router and self.local_router.is_ready:
            tiers.append("Tier 2 (Qwen3-4B MLX)")
        tiers.append("Tier 3 (Claude CLI)")
        print("\nRouting: %s" % " -> ".join(tiers))

        print("\nMCP servers:")
        for name in self.mcp.server_names:
            server = self.mcp.get_server(name)
            print("  - %s: %s" % (name, server.description if server else ""))
        print("\nProject directories:")
        for d in self.config.claude.project_directories:
            print("  - %s" % d)
        print("\nPress Enter to start speaking, Ctrl+C to exit")
        print("-" * 50)

        while self._running:
            try:
                self.state_manager.transition_to(AgentState.IDLE)

                print("\n[Press Enter to speak...]")
                await asyncio.get_event_loop().run_in_executor(None, input)

                if not self._running:
                    break

                continue_conversation = True
                while continue_conversation and self._running:
                    continue_conversation = await self.handle_turn(is_followup_answer=False)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print("\nError: %s" % e)
                if self.config.debug:
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(1)

        await self.shutdown()

    async def shutdown(self) -> None:
        """Clean up all resources"""
        print("\nShutting down Kat...")
        self._running = False

        await self.tts.interrupt()
        await self.audio_output.interrupt()

        if hasattr(self.tts, 'cleanup'):
            self.tts.cleanup()

        if self.deepgram_stt:
            await self.deepgram_stt.stop()
            await self.deepgram_stt.disconnect()
            self.deepgram_stt.cleanup()

        self.audio_input.cleanup()

        await self.mcp.shutdown()
        await self.session_store.shutdown()

        # Cleanup memory systems
        self.temporal_memory.cleanup_expired()
        self.temporal_memory.shutdown()
        self.long_term_memory.cleanup_expired()
        self.long_term_memory.shutdown()
        self.training_collector.shutdown()

        print("Goodbye!")

    def stop(self) -> None:
        """Stop the agent"""
        self._running = False
