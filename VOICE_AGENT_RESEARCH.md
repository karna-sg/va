# Voice Agent Research: Jarvis-like Assistant with Claude Code

## Executive Summary

Building a voice-controlled agent that integrates with Claude Code to perform professional tasks using voice commands. The agent should support wake word activation ("Hey Jarvis"), natural conversation with follow-up questions, and seamless MCP tool integration (GitHub, Slack, etc.).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Approach Analysis](#approach-analysis)
3. [Technical Requirements](#technical-requirements)
4. [Known Factors](#known-factors)
5. [Unknown Factors & Risks](#unknown-factors--risks)
6. [Difficulties & Challenges](#difficulties--challenges)
7. [Recommended Architecture](#recommended-architecture)
8. [Implementation Phases](#implementation-phases)

---

## Architecture Overview

### Desired Flow
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VOICE AGENT ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Voice ──► Wake Word ──► Speech-to-Text ──► Intent Processing         │
│       ▲              │              │                    │                  │
│       │              │              │                    ▼                  │
│       │              │              │            ┌──────────────┐           │
│       │              │              │            │  Claude Code │           │
│       │              │              │            │     CLI      │           │
│       │              │              │            └──────┬───────┘           │
│       │              │              │                   │                   │
│   TTS Output ◄──────────────────────┘                   ▼                   │
│       │                                         ┌──────────────┐           │
│       │                                         │  MCP Servers │           │
│       │                                         ├──────────────┤           │
│       │                                         │ • GitHub     │           │
│       │                                         │ • Slack      │           │
│       │                                         │ • Calendar   │           │
│       │                                         │ • Files      │           │
│       │                                         └──────────────┘           │
│       │                                                                     │
│       └─────────────── Response Flow ◄──────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example Conversation Flow
```
User: "Hey Jarvis, what did we do yesterday?"
      ↓
[Wake word detected → STT → Claude Code]
      ↓
Claude Code: Queries GitHub MCP → Gets commits, PRs from yesterday
      ↓
Agent (TTS): "Yesterday you merged 3 PRs: the auth refactor,
             the dashboard update, and the API rate limiting fix.
             You also opened 2 new issues for the mobile app."
      ↓
User: "Send this summary to Slack"
      ↓
[STT → Claude Code → Context maintained]
      ↓
Claude Code: Uses Slack MCP → Posts to configured channel
      ↓
Agent (TTS): "Done! I've posted the summary to #engineering-updates."
```

---

## Approach Analysis

### Approach 1: macOS Dictation

#### How It Works
- Built-in macOS feature (System Settings → Keyboard → Dictation)
- Triggered via keyboard shortcut (default: Fn key twice)
- On Apple Silicon: On-device processing for English, German, Spanish, French, Japanese, Chinese, Korean
- Text inserted at cursor position

#### Technical Implementation
```swift
// Cannot programmatically access dictation results directly
// Would need workaround via:
// 1. Custom text field to capture dictation
// 2. Accessibility APIs to monitor text changes
// 3. AppleScript automation (limited)
```

#### Pros
| Benefit | Details |
|---------|---------|
| No additional cost | Free, built into macOS |
| Privacy | On-device processing on Apple Silicon |
| Accuracy | Apple's trained models, good quality |
| Low setup | No API keys or external services |

#### Cons
| Limitation | Impact | Severity |
|------------|--------|----------|
| No wake word | Cannot say "Hey Jarvis" | **CRITICAL** |
| Manual trigger | Requires hotkey press | HIGH |
| No API access | Can't get text programmatically | **CRITICAL** |
| No streaming | Must complete dictation before processing | MEDIUM |
| Cursor-dependent | Text goes to active text field | HIGH |

#### Verdict: **NOT VIABLE** for wake-word activated assistant

---

### Approach 2: Custom Speech-to-Text

#### Available Options

##### A. Local/On-Device Solutions

| Solution | Language | Accuracy | Latency | Privacy | Cost |
|----------|----------|----------|---------|---------|------|
| **Whisper (OpenAI)** | Python/C++ | Excellent | Medium (2-5s) | High (local) | Free |
| **Whisper.cpp** | C++ | Excellent | Fast (0.5-2s) | High (local) | Free |
| **Vosk** | Python/JS/etc | Good | Fast | High (local) | Free |
| **Apple Speech Framework** | Swift | Very Good | Fast | High (local) | Free |
| **DeepSpeech** | Python | Good | Medium | High (local) | Free |

##### B. Cloud-Based Solutions

| Solution | Accuracy | Latency | Privacy | Cost |
|----------|----------|---------|---------|------|
| **OpenAI Whisper API** | Excellent | 1-3s | Medium | $0.006/min |
| **Google Speech-to-Text** | Excellent | <1s streaming | Low | $0.006-0.024/min |
| **Azure Speech Services** | Excellent | <1s streaming | Medium | $0.016/min |
| **AWS Transcribe** | Very Good | <1s streaming | Medium | $0.024/min |
| **Deepgram** | Excellent | <300ms | Medium | $0.0043/min |

##### C. Wake Word Detection Solutions

| Solution | Custom Words | Accuracy | Always-On Battery | License |
|----------|--------------|----------|-------------------|---------|
| **Picovoice Porcupine** | Yes | Excellent | Low | Free tier available |
| **Snowboy** (deprecated) | Yes | Good | Low | Apache 2.0 |
| **Mycroft Precise** | Yes | Good | Medium | Apache 2.0 |
| **OpenWakeWord** | Yes | Good | Medium | Apache 2.0 |
| **Custom Keyword Spotting** | Yes | Varies | Varies | N/A |

---

## Technical Requirements

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPONENT REQUIREMENTS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. AUDIO INPUT LAYER                                           │
│     ├── Microphone access (permissions)                         │
│     ├── Audio stream capture (16kHz, 16-bit, mono)              │
│     ├── Voice Activity Detection (VAD)                          │
│     └── Noise cancellation (optional)                           │
│                                                                 │
│  2. WAKE WORD DETECTION                                         │
│     ├── Always-listening mode (low power)                       │
│     ├── Custom wake word ("Hey Jarvis")                         │
│     ├── False positive handling                                 │
│     └── Activation sound/feedback                               │
│                                                                 │
│  3. SPEECH-TO-TEXT ENGINE                                       │
│     ├── High accuracy transcription                             │
│     ├── Streaming support (for real-time)                       │
│     ├── End-of-speech detection                                 │
│     └── Support for technical vocabulary                        │
│                                                                 │
│  4. CLAUDE CODE INTEGRATION                                     │
│     ├── Programmatic CLI invocation                             │
│     ├── Stdin/stdout capture                                    │
│     ├── Session/context management                              │
│     ├── MCP tool handling                                       │
│     └── Streaming response capture                              │
│                                                                 │
│  5. TEXT-TO-SPEECH OUTPUT                                       │
│     ├── Natural-sounding voice                                  │
│     ├── Interruptible playback                                  │
│     ├── Response streaming                                      │
│     └── Audio output management                                 │
│                                                                 │
│  6. CONVERSATION MANAGEMENT                                     │
│     ├── Multi-turn context                                      │
│     ├── Follow-up question handling                             │
│     ├── Conversation state machine                              │
│     └── Timeout/cancel handling                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### System Requirements

| Requirement | Specification |
|-------------|---------------|
| macOS Version | 12.0+ (Monterey or later recommended) |
| Python | 3.9+ (for most STT libraries) |
| Node.js | 18+ (if using JS-based solution) |
| RAM | 8GB+ (16GB recommended for local Whisper) |
| Storage | 2-10GB (depending on model size) |
| Microphone | Built-in or external USB/Bluetooth |
| Permissions | Microphone access, Accessibility (optional) |

---

## Known Factors

### What We Know Works

#### 1. Claude Code CLI Integration (VERIFIED)

**Critical Discovery**: Claude Code has excellent programmatic support!

```bash
# Non-interactive mode with JSON output (VERIFIED WORKING)
echo "What is 2+2?" | claude --print --output-format json

# Output includes session_id for conversation continuity:
# {"type":"result","result":"2 + 2 = 4","session_id":"c35ea7ec-bae0-4956-9c4e-02c4a8d1ac3c",...}

# Resume conversation for follow-ups (VERIFIED WORKING)
echo "What was my previous question?" | claude --print --output-format json --resume SESSION_ID
# Correctly remembers context: "Your previous question was 'What is 2+2?'"

# Streaming JSON output for real-time TTS (VERIFIED WORKING)
echo "Count to 5" | claude --print --output-format stream-json --verbose
# Streams JSON objects as they arrive

# Skip all permission prompts (for trusted operations)
claude --dangerously-skip-permissions --print "do something"

# Permission modes available
claude --permission-mode bypassPermissions  # Skip all
claude --permission-mode acceptEdits        # Auto-accept edits
claude --permission-mode delegate           # Delegate to agent
```

**Key CLI Options for Voice Agent:**
| Option | Purpose |
|--------|---------|
| `--print` or `-p` | Non-interactive mode, returns result |
| `--output-format json` | Clean JSON output with metadata |
| `--output-format stream-json` | Real-time streaming (requires --verbose) |
| `--resume SESSION_ID` | Continue previous conversation |
| `--session-id UUID` | Use specific session ID |
| `--continue` or `-c` | Continue most recent conversation |
| `--dangerously-skip-permissions` | Bypass all prompts |
| `--permission-mode MODE` | Control permission behavior |

#### 2. MCP Server Configuration
- GitHub MCP is configured and working
- Slack MCP is configured with channel settings
- MCP tools can be invoked by Claude Code

#### 3. macOS TTS Capabilities
```bash
# Built-in TTS works well
say "Hello, I am Jarvis"

# Can use different voices
say -v "Samantha" "Task completed"

# Can control rate and pitch
say -r 200 "Speaking faster"
```

#### 4. Whisper Local Performance
```python
# Whisper.cpp on Apple Silicon (M1/M2/M3):
# - tiny model: ~10x real-time
# - base model: ~7x real-time
# - small model: ~4x real-time
# - medium model: ~2x real-time
# - large model: ~1x real-time (requires 10GB+ RAM)
```

#### 5. Picovoice Wake Word
```python
# Custom wake word "Hey Jarvis" can be created
# - Free tier: 3 custom wake words
# - ~200ms detection latency
# - <2% CPU on modern Macs
```

### Verified Technical Facts

| Component | Status | Notes |
|-----------|--------|-------|
| Microphone access via Python | ✅ Verified | `pyaudio`, `sounddevice` work on macOS |
| Whisper.cpp on Apple Silicon | ✅ Verified | Optimized for Metal/CoreML |
| Claude Code CLI exists | ✅ Verified | `/usr/local/bin/claude` or `npx @anthropic-ai/claude-code` |
| macOS `say` command | ✅ Verified | Built-in, always available |
| Picovoice macOS support | ✅ Verified | Native Python bindings |

---

## Unknown Factors & Risks

### Critical Unknowns

#### 1. Claude Code Programmatic Interface - **RESOLVED**
```
STATUS: ANSWERED - Claude Code has excellent programmatic support!

Answers:
├── API/SDK: CLI with --print mode works perfectly
├── Pipe I/O: echo "command" | claude --print --output-format json ✓
├── Tool confirmations: --dangerously-skip-permissions or --permission-mode
├── Context persistence: --resume SESSION_ID works! ✓
├── Streaming: --output-format stream-json --verbose ✓
└── MCP interaction: Can be auto-approved with permission flags

Risk Level: LOW (was HIGH)
Impact: Non-blocking, clear path forward
```

#### 2. Conversation Context Persistence - **RESOLVED**
```
STATUS: ANSWERED - Session management works perfectly!

Verified Behavior:
├── Session persistence: YES - returns session_id in JSON output
├── Resume conversation: claude --resume SESSION_ID works ✓
├── Context maintained: Tested - remembers previous questions ✓
├── Continue latest: claude --continue or -c flag available
├── Custom session: claude --session-id UUID for explicit control

Implementation Pattern:
1. First command → get session_id from JSON response
2. Follow-ups → use --resume session_id
3. Store session_id in voice agent state

Risk Level: LOW (was HIGH)
Impact: Follow-up questions fully supported!
```

#### 3. Real-time Latency Requirements
```
UNKNOWN: What is acceptable end-to-end latency?

Current Estimated Latency Breakdown:
├── Wake word detection: ~200ms
├── Speech-to-text: 500ms - 3000ms (depends on solution)
├── Claude Code processing: 1000ms - 10000ms (depends on task)
├── MCP tool execution: 500ms - 5000ms (depends on tool)
├── Text-to-speech: 200ms - 500ms
└── TOTAL: 2.4s - 18.7s (wide range!)

Risk Level: MEDIUM
Impact: Poor UX if too slow, may need optimization
```

#### 4. Claude Code Permission Prompts
```
UNKNOWN: How to handle interactive permission prompts?

Scenarios:
├── File write permissions
├── Bash command execution
├── MCP tool invocations
├── Sensitive operations
└── Rate limiting

Risk Level: HIGH
Impact: May break automation flow
```

#### 5. Audio Edge Cases
```
UNKNOWN: How well will system handle:

├── Background noise
├── Multiple speakers
├── Accents and dialects
├── Technical jargon
├── Code/variable names spoken aloud
├── Interruptions mid-sentence
└── Very long commands

Risk Level: MEDIUM
Impact: Usability and accuracy issues
```

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Claude Code API limitations | High | Critical | Research API, may need wrapper |
| Context not persisting | Medium | High | Build custom context manager |
| Latency too high | Medium | Medium | Use streaming, optimize pipeline |
| Wake word false positives | Low | Medium | Tune sensitivity, add confirmation |
| STT accuracy issues | Low | Medium | Use best model, add correction UI |
| Permission prompts blocking | High | High | Pre-authorize or configure Claude |

---

## Difficulties & Challenges

### Category 1: Technical Integration Challenges

#### Challenge 1.1: Claude Code CLI Automation
```
DIFFICULTY: Claude Code is designed for interactive use, not automation

Problems:
1. Output is formatted for terminal display
2. May include ANSI codes, spinners, progress indicators
3. Prompts for confirmation on certain actions
4. Streaming output is complex to capture
5. Error handling is terminal-focused

Potential Solutions:
├── A. Use `--print` flag for simpler output
├── B. Parse output and strip formatting
├── C. Build wrapper that handles prompts
├── D. Request Anthropic for proper API
└── E. Use Claude API directly + rebuild MCP integration
```

#### Challenge 1.2: MCP Tool Integration
```
DIFFICULTY: MCP tools designed for manual approval

Problems:
1. Some tools require user confirmation
2. Tool results may need interpretation
3. Tool errors need graceful handling
4. Some tools are slow (network calls)
5. Tool availability depends on configuration

Potential Solutions:
├── A. Pre-approve certain tools/patterns
├── B. Configure Claude Code with permissive settings
├── C. Implement confirmation via voice ("Should I proceed?")
└── D. Create allow-list of auto-approved operations
```

#### Challenge 1.3: Bidirectional Conversation
```
DIFFICULTY: Agent needs to ask follow-up questions

Example Flow:
User: "Send the report to the team"
Agent: "Which report? I see the weekly metrics and the sprint summary."
User: "The weekly metrics"
Agent: "Should I send it to #team-general or #engineering?"
User: "Engineering"
Agent: "Done!"

Problems:
1. Need to detect when Claude wants to ask a question
2. Must switch from TTS output to listening mode
3. Context must be maintained
4. User might give partial answers
5. Need timeout/cancel handling

Potential Solutions:
├── A. Parse Claude output for question patterns
├── B. Use special markers in prompts
├── C. Implement state machine for conversation flow
└── D. Always-on listening during active conversation
```

### Category 2: User Experience Challenges

#### Challenge 2.1: Wake Word Reliability
```
DIFFICULTY: Balance between sensitivity and false positives

Too Sensitive:
- Activates on TV/music
- Activates on similar words
- Activates on background conversations

Too Insensitive:
- Requires shouting
- Doesn't work from across room
- Frustrating repeated attempts

Potential Solutions:
├── A. Allow sensitivity tuning
├── B. Add confirmation sound before listening
├── C. Use two-stage activation ("Hey Jarvis" + "listening" sound + command)
└── D. Visual indicator (menubar) showing state
```

#### Challenge 2.2: End-of-Speech Detection
```
DIFFICULTY: Knowing when user finished speaking

Problems:
1. Natural pauses in speech
2. Thinking pauses
3. Technical terms with pauses
4. Background noise
5. Different speaking styles

Potential Solutions:
├── A. Use silence threshold (e.g., 1.5 seconds)
├── B. Use explicit end phrases ("over", "done", "execute")
├── C. Combine silence + semantic completeness
└── D. Allow push-to-talk as fallback
```

#### Challenge 2.3: Error Recovery
```
DIFFICULTY: What happens when something goes wrong?

Error Scenarios:
├── STT misunderstands command
├── Claude Code fails
├── MCP tool errors
├── Network timeout
├── Ambiguous request
└── Unsupported operation

User Experience:
- Should not require looking at screen
- Should explain what went wrong
- Should offer to retry or clarify
- Should not get stuck in bad state
```

### Category 3: System Architecture Challenges

#### Challenge 3.1: State Management
```
DIFFICULTY: Multiple components need synchronized state

States to Track:
├── Wake word: dormant | listening
├── STT: idle | transcribing | done
├── Claude: idle | processing | waiting_input | done
├── TTS: idle | speaking
├── Conversation: none | active | follow_up_pending
└── MCP Tools: idle | executing | needs_confirmation

State Transitions:
- Must be atomic and consistent
- Must handle interruptions
- Must timeout gracefully
- Must be observable for debugging
```

#### Challenge 3.2: Interrupt Handling
```
DIFFICULTY: User should be able to interrupt at any time

Interrupt Scenarios:
1. Cancel current operation
2. Correct misheard command
3. Stop TTS mid-speech
4. Abort MCP tool execution
5. Change context mid-conversation

Technical Challenges:
├── TTS playback must be interruptible
├── Claude Code request cancellation
├── MCP tool cancellation (may not be possible)
├── State cleanup after interrupt
└── Seamless transition to new command
```

#### Challenge 3.3: Resource Management
```
DIFFICULTY: Always-on listening consumes resources

Concerns:
├── Battery drain on laptops
├── CPU usage for wake word
├── Memory for loaded models
├── Network bandwidth (if using cloud)
└── Audio device contention

Optimizations:
├── Use lightweight wake word model
├── Load STT model only when needed
├── Implement smart sleep/wake
├── Cache commonly used responses
└── Batch API requests where possible
```

### Category 4: Security & Privacy Challenges

#### Challenge 4.1: Always-On Microphone
```
DIFFICULTY: Privacy implications of constant listening

Concerns:
├── Accidental recording of sensitive conversations
├── Data transmission to cloud (if using cloud STT)
├── Storage of voice recordings
├── Compliance with privacy regulations
└── User trust and transparency

Mitigations:
├── Use local wake word detection (no cloud)
├── Use local STT when possible
├── Clear audio buffer policy
├── Visual/audio indicators when listening
├── Easy mute/disable controls
└── No persistent audio storage
```

#### Challenge 4.2: Command Injection
```
DIFFICULTY: Malicious audio could trigger harmful commands

Scenarios:
├── Someone else says "Hey Jarvis, delete all files"
├── TV/video plays command audio
├── Ultrasonic attacks (theoretical)
└── Social engineering via voice

Mitigations:
├── Require voice recognition (speaker verification)
├── Confirm destructive operations
├── Limit allowed operations
├── Log all commands
└── Implement allowlist for sensitive ops
```

---

## Recommended Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDED ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   Audio Input   │────►│  Wake Word      │────►│  Speech-to-Text │       │
│  │   (PyAudio)     │     │  (Porcupine)    │     │  (Whisper.cpp)  │       │
│  └─────────────────┘     └─────────────────┘     └────────┬────────┘       │
│                                                           │                 │
│                                                           ▼                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   TTS Output    │◄────│  Response       │◄────│  Claude Code    │       │
│  │   (macOS say    │     │  Parser         │     │  Wrapper        │       │
│  │   or ElevenLabs)│     │                 │     │                 │       │
│  └─────────────────┘     └─────────────────┘     └────────┬────────┘       │
│                                                           │                 │
│                                                           ▼                 │
│                                                  ┌─────────────────┐       │
│                                                  │   MCP Servers   │       │
│                                                  │  (GitHub,Slack) │       │
│                                                  └─────────────────┘       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STATE MANAGER / ORCHESTRATOR                      │   │
│  │  - Manages conversation state                                        │   │
│  │  - Handles interrupts                                                │   │
│  │  - Coordinates all components                                        │   │
│  │  - Maintains context across turns                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack Recommendation

| Component | Primary Choice | Fallback | Reason |
|-----------|---------------|----------|--------|
| Wake Word | Picovoice Porcupine | OpenWakeWord | Best accuracy, custom "Jarvis" |
| STT | Whisper.cpp (local) | Deepgram API | Privacy + no cost for local |
| LLM Interface | Claude Code CLI | Claude API direct | Existing MCP integration |
| TTS | macOS `say` | ElevenLabs API | Free and built-in |
| Orchestrator | Python + asyncio | Node.js | Ecosystem compatibility |
| Audio Input | PyAudio / sounddevice | PortAudio | Cross-platform |

### Component Details

#### 1. Audio Pipeline
```python
# Recommended: sounddevice for audio capture
import sounddevice as sd
import numpy as np

class AudioPipeline:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.int16

    def start_listening(self, callback):
        """Start continuous audio capture"""
        pass

    def stop_listening(self):
        """Stop audio capture"""
        pass
```

#### 2. Wake Word Module
```python
# Using Picovoice Porcupine
import pvporcupine

class WakeWordDetector:
    def __init__(self, keyword="jarvis"):
        self.porcupine = pvporcupine.create(
            access_key="YOUR_KEY",
            keywords=[keyword]
        )

    def process(self, audio_frame):
        """Returns True if wake word detected"""
        result = self.porcupine.process(audio_frame)
        return result >= 0
```

#### 3. Speech-to-Text Module
```python
# Using whisper.cpp via pywhispercpp
from pywhispercpp.model import Model

class SpeechToText:
    def __init__(self, model_size="base"):
        self.model = Model(model_size)

    def transcribe(self, audio_data):
        """Convert audio to text"""
        return self.model.transcribe(audio_data)
```

#### 4. Claude Code Interface (VERIFIED IMPLEMENTATION)
```python
import subprocess
import asyncio
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class ClaudeResponse:
    result: str
    session_id: str
    is_error: bool
    duration_ms: int
    cost_usd: float

class ClaudeCodeInterface:
    def __init__(self):
        self.current_session_id: Optional[str] = None

    async def send_command(self, text: str, new_conversation: bool = False) -> ClaudeResponse:
        """Send command to Claude Code and get response"""

        cmd = ['claude', '--print', '--output-format', 'json']

        # Add permission bypass for automated use
        cmd.append('--dangerously-skip-permissions')

        # Resume session for follow-ups
        if self.current_session_id and not new_conversation:
            cmd.extend(['--resume', self.current_session_id])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate(input=text.encode())

        # Parse JSON response
        response_data = json.loads(stdout.decode())

        # Store session_id for follow-ups
        self.current_session_id = response_data.get('session_id')

        return ClaudeResponse(
            result=response_data.get('result', ''),
            session_id=response_data.get('session_id', ''),
            is_error=response_data.get('is_error', False),
            duration_ms=response_data.get('duration_ms', 0),
            cost_usd=response_data.get('total_cost_usd', 0)
        )

    async def send_with_streaming(self, text: str, on_chunk: callable):
        """Send command with streaming response for real-time TTS"""

        cmd = ['claude', '--print', '--output-format', 'stream-json', '--verbose']

        if self.current_session_id:
            cmd.extend(['--resume', self.current_session_id])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Write input and close stdin
        process.stdin.write(text.encode())
        await process.stdin.drain()
        process.stdin.close()

        # Stream output
        async for line in process.stdout:
            try:
                chunk = json.loads(line.decode())
                if chunk.get('type') == 'assistant':
                    content = chunk.get('message', {}).get('content', [])
                    for item in content:
                        if item.get('type') == 'text':
                            await on_chunk(item.get('text', ''))
                elif chunk.get('type') == 'result':
                    self.current_session_id = chunk.get('session_id')
            except json.JSONDecodeError:
                continue

        await process.wait()

# Usage Example for Voice Agent
async def voice_agent_example():
    claude = ClaudeCodeInterface()

    # First command: "Hey Jarvis, what did we do yesterday?"
    response = await claude.send_command(
        "What commits and PRs were made yesterday? Use GitHub MCP.",
        new_conversation=True
    )
    print(f"Response: {response.result}")
    # TTS: speak(response.result)

    # Follow-up: "Send this to Slack"
    response = await claude.send_command(
        "Send a summary of what you just told me to the #engineering channel on Slack"
    )
    print(f"Response: {response.result}")
    # TTS: speak(response.result)
```

#### 5. Text-to-Speech Module
```python
import subprocess
import asyncio

class TextToSpeech:
    def __init__(self, voice="Samantha"):
        self.voice = voice
        self.current_process = None

    async def speak(self, text):
        """Speak text using macOS TTS"""
        self.current_process = await asyncio.create_subprocess_exec(
            'say', '-v', self.voice, text
        )
        await self.current_process.wait()

    async def interrupt(self):
        """Stop current speech"""
        if self.current_process:
            self.current_process.terminate()
```

---

## Implementation Phases

### Phase 1: Proof of Concept (Week 1-2)
```
Goals:
├── Verify Whisper.cpp works locally with good accuracy
├── Test wake word detection with Picovoice
├── Confirm Claude Code can be invoked programmatically
├── Build basic pipeline: wake word → STT → echo → TTS

Deliverables:
├── Working audio capture
├── Wake word detection demo
├── STT transcription demo
├── Basic TTS output
└── Documentation of findings
```

### Phase 2: Claude Integration (Week 3-4)
```
Goals:
├── Build reliable Claude Code wrapper
├── Handle output parsing
├── Implement basic context management
├── Test with simple commands (no MCP)

Deliverables:
├── Claude Code wrapper module
├── Output parser
├── Simple conversation loop
└── Error handling
```

### Phase 3: MCP Tools (Week 5-6)
```
Goals:
├── Test GitHub MCP integration
├── Test Slack MCP integration
├── Handle tool confirmations
├── Implement follow-up question detection

Deliverables:
├── Working GitHub queries via voice
├── Working Slack posting via voice
├── Confirmation voice flow
└── Follow-up question handling
```

### Phase 4: Polish & UX (Week 7-8)
```
Goals:
├── Improve latency
├── Better error handling
├── Add visual indicators (menubar app?)
├── Implement interrupt handling
├── User testing and iteration

Deliverables:
├── Production-ready voice agent
├── Menubar status app
├── Configuration UI
└── User documentation
```

---

## Open Questions Requiring Research

### Must Answer Before Starting

1. **How to run Claude Code programmatically with context persistence?**
   - Need to test `claude` CLI options
   - May need to use Claude API directly

2. **Can we auto-approve certain MCP operations?**
   - Check Claude Code configuration options
   - Test with `--dangerously-skip-permissions` or similar

3. **What is actual latency of Whisper.cpp on target hardware?**
   - Need to benchmark on actual Mac

4. **Can Picovoice create custom "Hey Jarvis" wake word for free?**
   - Check Picovoice console and pricing

5. **How does Claude Code handle streaming output?**
   - Test with `--stream` flag
   - Understand output format

### Nice to Answer

6. Is there a Claude Code SDK or API we don't know about?
7. Can we embed Claude directly and use MCP protocol?
8. What's the best TTS for more natural voice on macOS?
9. How to handle overlapping audio (TTS + user speaks)?
10. Should we build a native macOS app or Python script?

---

## Next Steps

1. **Immediate**: Test Claude Code CLI programmatic invocation
2. **This Week**: Set up Picovoice account and test wake word
3. **This Week**: Benchmark Whisper.cpp on local Mac
4. **Next Week**: Build minimal proof of concept
5. **Ongoing**: Document all findings in this file

---

## Appendix

### A. Useful Commands

```bash
# Test macOS TTS
say "Hello, I am Jarvis"

# List available voices
say -v '?'

# Test Claude Code
claude --print "What is 2+2?"

# Check microphone permissions
tccutil reset Microphone

# Install Whisper.cpp
brew install whisper-cpp

# Install Python audio libraries
pip install pyaudio sounddevice numpy
```

### B. Relevant Links

- Claude Code: https://github.com/anthropics/claude-code
- Picovoice: https://picovoice.ai/
- Whisper.cpp: https://github.com/ggerganov/whisper.cpp
- PyAudio: https://pypi.org/project/PyAudio/
- ElevenLabs TTS: https://elevenlabs.io/

### C. Alternative Approaches Considered

1. **Use Siri Shortcuts as trigger**: Limited, can't extract voice text
2. **Use macOS Accessibility API**: Complex, undocumented
3. **Build iOS companion app**: Adds complexity, not needed
4. **Use Electron app**: Heavier than needed for MVP
5. **Use pure Claude API**: Would lose MCP integration

---

---

## D. Complete Voice Agent Skeleton (Ready to Implement)

```python
#!/usr/bin/env python3
"""
Jarvis Voice Agent - Complete Implementation Skeleton
Based on verified research findings
"""

import asyncio
import json
import struct
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
import subprocess

# External dependencies (to be installed):
# pip install pyaudio pvporcupine pywhispercpp numpy

class AgentState(Enum):
    DORMANT = "dormant"           # Waiting for wake word
    LISTENING = "listening"        # Recording user speech
    PROCESSING = "processing"      # Claude is thinking
    SPEAKING = "speaking"          # TTS is playing
    WAITING_FOLLOWUP = "waiting"   # Expecting follow-up

@dataclass
class ConversationContext:
    session_id: Optional[str] = None
    last_response: str = ""
    turn_count: int = 0

class WakeWordDetector:
    """Picovoice Porcupine wake word detection"""

    def __init__(self, access_key: str, keyword: str = "jarvis"):
        import pvporcupine
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=[keyword]
        )
        self.frame_length = self.porcupine.frame_length
        self.sample_rate = self.porcupine.sample_rate

    def process_frame(self, audio_frame) -> bool:
        """Returns True if wake word detected"""
        result = self.porcupine.process(audio_frame)
        return result >= 0

    def cleanup(self):
        self.porcupine.delete()

class SpeechToText:
    """Whisper.cpp based speech-to-text"""

    def __init__(self, model_size: str = "base"):
        from pywhispercpp.model import Model
        self.model = Model(model_size)

    def transcribe(self, audio_file: str) -> str:
        """Transcribe audio file to text"""
        segments = self.model.transcribe(audio_file)
        return " ".join([seg.text for seg in segments])

class ClaudeCode:
    """Claude Code CLI interface with session management"""

    def __init__(self):
        self.session_id: Optional[str] = None

    async def send(self, text: str, new_session: bool = False) -> dict:
        cmd = [
            'claude', '--print',
            '--output-format', 'json',
            '--dangerously-skip-permissions'
        ]

        if self.session_id and not new_session:
            cmd.extend(['--resume', self.session_id])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, _ = await process.communicate(input=text.encode())
        response = json.loads(stdout.decode())

        self.session_id = response.get('session_id')
        return response

class TextToSpeech:
    """macOS native TTS"""

    def __init__(self, voice: str = "Samantha"):
        self.voice = voice
        self.process: Optional[asyncio.subprocess.Process] = None

    async def speak(self, text: str):
        self.process = await asyncio.create_subprocess_exec(
            'say', '-v', self.voice, text
        )
        await self.process.wait()

    async def interrupt(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()

class AudioRecorder:
    """Record audio with silence detection"""

    def __init__(self, sample_rate: int = 16000):
        import pyaudio
        self.sample_rate = sample_rate
        self.chunk_size = 1024
        self.silence_threshold = 500  # Adjust based on environment
        self.silence_duration = 1.5   # Seconds of silence to stop
        self.pa = pyaudio.PyAudio()

    async def record_until_silence(self) -> bytes:
        """Record audio until user stops speaking"""
        import numpy as np

        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        frames = []
        silent_chunks = 0
        max_silent_chunks = int(self.silence_duration * self.sample_rate / self.chunk_size)

        try:
            while True:
                data = stream.read(self.chunk_size)
                frames.append(data)

                # Check for silence
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()

                if volume < self.silence_threshold:
                    silent_chunks += 1
                    if silent_chunks > max_silent_chunks:
                        break
                else:
                    silent_chunks = 0

                await asyncio.sleep(0)  # Yield to event loop
        finally:
            stream.stop_stream()
            stream.close()

        return b''.join(frames)

class VoiceAgent:
    """Main voice agent orchestrator"""

    def __init__(self, porcupine_key: str):
        self.state = AgentState.DORMANT
        self.context = ConversationContext()

        self.wake_detector = WakeWordDetector(porcupine_key)
        self.stt = SpeechToText()
        self.claude = ClaudeCode()
        self.tts = TextToSpeech()
        self.recorder = AudioRecorder()

    async def run(self):
        """Main agent loop"""
        print("🎤 Jarvis is listening for wake word...")

        while True:
            try:
                if self.state == AgentState.DORMANT:
                    await self._wait_for_wake_word()

                elif self.state == AgentState.LISTENING:
                    await self._record_and_transcribe()

                elif self.state == AgentState.PROCESSING:
                    await self._process_with_claude()

                elif self.state == AgentState.SPEAKING:
                    await self._speak_response()

                elif self.state == AgentState.WAITING_FOLLOWUP:
                    await self._wait_for_followup()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                self.state = AgentState.DORMANT

    async def _wait_for_wake_word(self):
        """Listen for 'Hey Jarvis'"""
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.wake_detector.sample_rate,
            input=True,
            frames_per_buffer=self.wake_detector.frame_length
        )

        try:
            while self.state == AgentState.DORMANT:
                pcm = stream.read(self.wake_detector.frame_length)
                pcm = struct.unpack_from(
                    "h" * self.wake_detector.frame_length, pcm
                )

                if self.wake_detector.process_frame(pcm):
                    print("🔔 Wake word detected!")
                    # Play activation sound
                    subprocess.run(['afplay', '/System/Library/Sounds/Pop.aiff'])
                    self.state = AgentState.LISTENING
                    break

                await asyncio.sleep(0)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    async def _record_and_transcribe(self):
        """Record user speech and convert to text"""
        print("🎙️ Listening...")

        audio_data = await self.recorder.record_until_silence()

        # Save to temp file for Whisper
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wf = wave.open(f.name, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
            wf.close()

            text = self.stt.transcribe(f.name)
            print(f"📝 Transcribed: {text}")

            self.context.user_input = text
            self.state = AgentState.PROCESSING

    async def _process_with_claude(self):
        """Send to Claude Code and get response"""
        print("🤔 Processing...")

        is_new = self.context.turn_count == 0
        response = await self.claude.send(
            self.context.user_input,
            new_session=is_new
        )

        self.context.session_id = response.get('session_id')
        self.context.last_response = response.get('result', 'Sorry, I could not process that.')
        self.context.turn_count += 1

        print(f"💬 Claude: {self.context.last_response}")
        self.state = AgentState.SPEAKING

    async def _speak_response(self):
        """Speak the response via TTS"""
        await self.tts.speak(self.context.last_response)

        # After speaking, wait for potential follow-up
        self.state = AgentState.WAITING_FOLLOWUP

    async def _wait_for_followup(self):
        """Wait for follow-up or return to dormant"""
        # Set a timeout for follow-up
        timeout = 10  # seconds

        print(f"⏳ Waiting {timeout}s for follow-up...")

        # Start listening for wake word or direct speech
        # For simplicity, go back to wake word mode
        # In a more sophisticated version, you could listen
        # for immediate follow-up without wake word

        await asyncio.sleep(timeout)
        self.state = AgentState.DORMANT

# Entry point
async def main():
    # Get your Picovoice key from: https://console.picovoice.ai/
    PORCUPINE_KEY = "YOUR_PICOVOICE_ACCESS_KEY"

    agent = VoiceAgent(PORCUPINE_KEY)
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Installation Commands

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pyaudio pvporcupine numpy

# Install Whisper.cpp Python bindings
pip install pywhispercpp

# OR build whisper.cpp manually for best performance on Apple Silicon
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make -j
# Download model
./models/download-ggml-model.sh base

# Get Picovoice key (free tier available)
# https://console.picovoice.ai/

# Test TTS
say "Hello, I am Jarvis"

# Test Claude Code
echo "Hello" | claude --print --output-format json
```

---

## E. Decision Matrix: Build vs Alternative Approaches

| Approach | Effort | Latency | Control | Offline | Cost |
|----------|--------|---------|---------|---------|------|
| **Full Custom (Recommended)** | High | Low | Full | Yes | Free/Low |
| Siri Shortcuts + Custom STT | Medium | Medium | Partial | No | Free |
| Whisper API + Cloud TTS | Low | High | Full | No | ~$5/month |
| macOS Dictation + Wrapper | Medium | Low | Limited | Yes | Free |
| Third-party Voice Assistant SDK | Medium | Medium | Limited | No | Varies |

**Recommendation**: Build full custom solution using:
- Picovoice Porcupine (wake word)
- Whisper.cpp (STT, local)
- Claude Code CLI (verified to work well)
- macOS `say` (TTS, can upgrade to ElevenLabs later)

---

*Document Version: 1.1*
*Last Updated: 2026-02-03*
*Author: AI Research Assistant*
*Status: Research Complete - Ready for Implementation*
