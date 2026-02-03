# Voice Agent POC Implementation Plan

## Answers to Key Questions

---

### Q1: Claude Code Programmatic Usage with Context Persistence

**STATUS: FULLY VERIFIED**

| Feature | Command | Verified |
|---------|---------|----------|
| Non-interactive mode | `echo "prompt" \| claude --print` | Yes |
| JSON output | `--output-format json` | Yes |
| Get session ID | Returns `session_id` in JSON | Yes |
| Resume session | `--resume SESSION_ID` | Yes |
| Continue latest | `--continue` or `-c` | Yes |
| Streaming | `--output-format stream-json --verbose` | Yes |

**Tested Example:**
```bash
# First command - get session_id
$ echo "Remember: 42" | claude --print --output-format json
# Returns: {"session_id":"24c9703e-63d7-428e-8b16-c6f6afddb89f", "result":"I've noted..."}

# Follow-up - resume with session_id
$ echo "What number?" | claude --print --output-format json --resume 24c9703e-63d7-428e-8b16-c6f6afddb89f
# Returns: {"result":"42", ...}  # CONTEXT MAINTAINED!
```

**Implementation Pattern:**
```python
import json
import subprocess

class ClaudeSession:
    def __init__(self):
        self.session_id = None

    async def query(self, text: str) -> str:
        cmd = ['claude', '--print', '--output-format', 'json']
        if self.session_id:
            cmd.extend(['--resume', self.session_id])

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=PIPE, stdout=PIPE
        )
        stdout, _ = await proc.communicate(text.encode())
        result = json.loads(stdout)

        self.session_id = result['session_id']  # Save for follow-ups
        return result['result']
```

---

### Q2: Auto-Approve MCP Operations

**STATUS: FULLY VERIFIED**

| Method | Command | Use Case |
|--------|---------|----------|
| Bypass all | `--dangerously-skip-permissions` | Full automation |
| Bypass via mode | `--permission-mode bypassPermissions` | Same as above |
| Accept edits only | `--permission-mode acceptEdits` | File changes |
| Don't ask | `--permission-mode dontAsk` | Silent mode |
| Allow specific tools | `--allowedTools "Bash(git:*) Edit"` | Fine-grained |

**Tested Example:**
```bash
# Create file WITHOUT any prompts
$ echo "Create /tmp/test.txt with 'hello'" | claude --print \
    --output-format json --permission-mode bypassPermissions
# File created successfully - NO PROMPTS!
```

**Recommendation for Voice Agent:**
```bash
claude --print --output-format json \
    --permission-mode bypassPermissions \
    --resume $SESSION_ID
```

---

### Q3: Whisper.cpp Latency on M4 Mac

**STATUS: BENCHMARKED (via research - actual M4 benchmarks)**

| Model | Load Time | Transcribe 10s Audio | Accuracy | Real-time Factor |
|-------|-----------|---------------------|----------|------------------|
| **tiny** | 0.24s | 0.37s | 99.2% | **27x faster** |
| **base** | 0.43s | 0.54s | 100% | **18x faster** |
| **small** | 1.04s | 1.44s | 100% | **7x faster** |
| medium | ~2s | ~3s | 100% | ~3x faster |
| large-v3-turbo | ~3s | ~4s | 100% | ~2x faster |

**Your Hardware:** MacBook Air M4, 16GB RAM

**Recommendation:** Use **base** model for best accuracy/speed balance
- 10 seconds of speech transcribed in **0.54 seconds**
- 100% accuracy
- Model loads in under 0.5 seconds

**Expected End-to-End STT Latency:** ~600-800ms for typical commands

---

### Q4: Picovoice Custom Wake Word

**STATUS: FREE TIER CONFIRMED**

| Feature | Free Tier | Paid ($6000+/year) |
|---------|-----------|-------------------|
| Monthly Active Users | 1 | 100+ |
| Wake Word Models | 1/month | 10/month |
| Custom Wake Words | **Yes - "Hey Jarvis"** | Yes |
| Training | Self-service console | Self-service |
| Platforms | All (macOS, iOS, etc.) | All |
| Commercial Use | **No** | Yes |

**How to Create Custom Wake Word:**
1. Sign up at [console.picovoice.ai](https://console.picovoice.ai)
2. Navigate to Porcupine section
3. Type "Hey Jarvis" in the text field
4. Model trains in **seconds** (transfer learning)
5. Download model file for your platform

**No coding required for training!**

---

### Q5: Claude Code Streaming Output

**STATUS: FULLY VERIFIED**

**Command:**
```bash
echo "prompt" | claude --print --output-format stream-json --verbose
```

**Output Format (3 JSON lines):**
```json
// Line 1: System init
{"type":"system","subtype":"init","session_id":"xxx","tools":[...]}

// Line 2: Assistant response (THE ACTUAL CONTENT)
{"type":"assistant","message":{"content":[{"type":"text","text":"response here"}]}}

// Line 3: Final result
{"type":"result","subtype":"success","result":"response here","session_id":"xxx"}
```

**Implementation for Real-time TTS:**
```python
async def stream_response(prompt: str, on_text: Callable):
    proc = await asyncio.create_subprocess_exec(
        'claude', '--print', '--output-format', 'stream-json', '--verbose',
        stdin=PIPE, stdout=PIPE
    )
    proc.stdin.write(prompt.encode())
    await proc.stdin.drain()
    proc.stdin.close()

    async for line in proc.stdout:
        chunk = json.loads(line)
        if chunk['type'] == 'assistant':
            text = chunk['message']['content'][0]['text']
            await on_text(text)  # Send to TTS immediately
```

---

### Q6: Claude Code SDK/API

**STATUS: YES - AGENT SDK EXISTS**

| Package | Status | Use Case |
|---------|--------|----------|
| `@anthropic-ai/claude-code` | **Deprecated** | Legacy |
| `@anthropic-ai/claude-agent-sdk` | **Current** | Building agents |
| Python SDK | Available | `pip install anthropic` |

**Claude Agent SDK Features:**
- Programmatic access to Claude Code capabilities
- Custom tool definitions
- Session management
- MCP integration
- Hooks for custom behavior

**Installation:**
```bash
npm install @anthropic-ai/claude-agent-sdk
# OR
pip install anthropic
```

**For POC:** CLI approach is simpler and verified working. Consider SDK for production.

---

### Q7: Embed Claude with MCP Directly

**STATUS: YES - MCP SDKs AVAILABLE**

You can use MCP servers directly without Claude Code:

| SDK | Language | Installation |
|-----|----------|-------------|
| Official | TypeScript | `npm install @modelcontextprotocol/sdk` |
| Official | Python | `pip install mcp` |
| FastMCP | Python | `pip install fastmcp` |

**Direct MCP Client Example:**
```python
from mcp import Client, StdioClientTransport

# Connect to MCP server directly
transport = StdioClientTransport(command="your-mcp-server")
client = Client(transport)
await client.connect()

# Call MCP tools
result = await client.call_tool("github_get_commits", {"since": "yesterday"})
```

**For POC:** Use Claude Code CLI with existing MCP config. Faster to implement.

---

### Q8: Best TTS for Natural Voice on macOS

**STATUS: RESEARCHED**

| Option | Quality | Latency | Cost | Offline |
|--------|---------|---------|------|---------|
| macOS `say` (Zoe/Anna Premium) | Good | Instant | Free | Yes |
| **ElevenLabs** | Excellent | 150ms TTFA | $99/mo | No |
| OpenAI TTS | Very Good | 200ms TTFA | $15/1M chars | No |
| OpenAI TTS Mini | Good | ~100ms | $0.60/1M chars | No |

**Recommendation for POC:**
1. **Start with:** macOS `say -v Samantha` (free, instant, offline)
2. **Upgrade to:** ElevenLabs or OpenAI TTS for production

**macOS Premium Voices Setup:**
```
System Settings → Accessibility → Spoken Content → System Voice → Manage Voices
Download: Zoe (Premium), Samantha (Enhanced)
```

---

### Q9: Handle Overlapping Audio (Barge-in)

**STATUS: RESEARCHED**

**Key Techniques:**

1. **Echo Cancellation (AEC)**
   - Prevents TTS audio from triggering STT
   - Required for duplex operation

2. **Voice Activity Detection (VAD)**
   - 10-20ms frame processing
   - Silero VAD recommended (fast, accurate)
   - 250-350ms hangover window

3. **Barge-in Detection**
   - Monitor mic during TTS playback
   - Stop TTS on sustained user speech (>200ms)
   - Double-talk detection to prevent false triggers

**Simple POC Implementation:**
```python
class AudioManager:
    def __init__(self):
        self.tts_playing = False
        self.vad = SileroVAD()

    async def play_tts(self, text: str):
        self.tts_playing = True
        process = await asyncio.create_subprocess_exec('say', text)

        # Monitor for barge-in
        while process.returncode is None:
            if self.detect_user_speech():
                process.terminate()  # Stop TTS
                return "interrupted"
            await asyncio.sleep(0.05)

        self.tts_playing = False
        return "completed"

    def detect_user_speech(self) -> bool:
        # Use VAD to detect speech during TTS
        audio_frame = self.get_mic_frame()
        return self.vad.is_speech(audio_frame)
```

**Latency Target:** Sub-100ms barge-in detection

---

### Q10: Native macOS App vs Python Script

**STATUS: RECOMMENDATION**

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Python Script** | Fast dev, easy debugging, rich ML libs | Packaging harder, permissions | **POC** |
| Native Swift App | Better performance, native audio APIs, menubar | Slower dev, different skills | Production |
| Electron App | Cross-platform, JS ecosystem | Heavy, overkill | Not recommended |

**POC Recommendation: Python Script**

**Reasons:**
1. Whisper.cpp has Python bindings (`pywhispercpp`)
2. Picovoice has Python SDK
3. Easy to iterate and debug
4. Can call `say` command for TTS
5. Rich async support with `asyncio`

**Production Path:**
1. POC in Python → validate approach
2. If successful, consider Swift wrapper for:
   - Menubar app with status indicator
   - Better audio permissions handling
   - Native notifications
   - Lower memory footprint

---

## POC Implementation Plan

### Phase 1: Core Pipeline (Days 1-3)

```
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 1: CORE PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Day 1: Audio Foundation                                        │
│  ├── [ ] Install dependencies (pyaudio, numpy)                  │
│  ├── [ ] Test microphone capture                                │
│  ├── [ ] Implement basic audio recording                        │
│  └── [ ] Test macOS `say` command for TTS                       │
│                                                                 │
│  Day 2: Speech-to-Text                                          │
│  ├── [ ] Install whisper.cpp / pywhispercpp                     │
│  ├── [ ] Download base model                                    │
│  ├── [ ] Implement transcription function                       │
│  ├── [ ] Benchmark on your M4 Mac                               │
│  └── [ ] Add silence detection for end-of-speech                │
│                                                                 │
│  Day 3: Claude Integration                                      │
│  ├── [ ] Implement ClaudeSession class                          │
│  ├── [ ] Test JSON output parsing                               │
│  ├── [ ] Verify session resumption                              │
│  ├── [ ] Test with permission bypass                            │
│  └── [ ] End-to-end test: Audio → STT → Claude → TTS            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Deliverable:** Working pipeline - press key → speak → Claude responds → TTS speaks

### Phase 2: Wake Word (Days 4-5)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: WAKE WORD                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Day 4: Picovoice Setup                                         │
│  ├── [ ] Create Picovoice account (free)                        │
│  ├── [ ] Generate "Hey Jarvis" model in console                 │
│  ├── [ ] Download model file for macOS                          │
│  ├── [ ] Install pvporcupine Python package                     │
│  └── [ ] Test wake word detection standalone                    │
│                                                                 │
│  Day 5: Integration                                             │
│  ├── [ ] Integrate wake word with audio capture                 │
│  ├── [ ] Add activation sound (Pop.aiff)                        │
│  ├── [ ] Implement state machine (dormant/listening)            │
│  └── [ ] Test full flow: "Hey Jarvis" → command → response      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Deliverable:** "Hey Jarvis, what time is it?" → works completely hands-free

### Phase 3: MCP Integration (Days 6-7)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: MCP TOOLS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Day 6: GitHub MCP                                              │
│  ├── [ ] Verify GitHub MCP is configured                        │
│  ├── [ ] Test: "What did I commit yesterday?"                   │
│  ├── [ ] Test: "Show my open PRs"                               │
│  └── [ ] Verify voice response is natural                       │
│                                                                 │
│  Day 7: Slack MCP + Multi-turn                                  │
│  ├── [ ] Verify Slack MCP is configured                         │
│  ├── [ ] Test: "Send summary to Slack"                          │
│  ├── [ ] Test multi-turn: query → follow-up                     │
│  └── [ ] Implement conversation timeout                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Deliverable:** Full example flow works:
1. "Hey Jarvis, what did we do yesterday?"
2. Claude fetches from GitHub, speaks summary
3. "Send this to Slack"
4. Claude posts to configured channel

### Phase 4: Polish (Days 8-10)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 4: POLISH                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Day 8: Error Handling                                          │
│  ├── [ ] Handle Claude errors gracefully                        │
│  ├── [ ] Handle STT failures                                    │
│  ├── [ ] Add timeout handling                                   │
│  └── [ ] Implement retry logic                                  │
│                                                                 │
│  Day 9: UX Improvements                                         │
│  ├── [ ] Add audio feedback (beeps, sounds)                     │
│  ├── [ ] Improve silence detection thresholds                   │
│  ├── [ ] Add basic barge-in support                             │
│  └── [ ] Test in noisy environment                              │
│                                                                 │
│  Day 10: Documentation & Demo                                   │
│  ├── [ ] Write setup instructions                               │
│  ├── [ ] Record demo video                                      │
│  ├── [ ] Document known limitations                             │
│  └── [ ] Plan production improvements                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Deliverable:** Polished POC ready for demonstration

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           POC ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          ┌──────────────┐                                   │
│                          │   Main Loop  │                                   │
│                          │  (asyncio)   │                                   │
│                          └──────┬───────┘                                   │
│                                 │                                           │
│         ┌───────────────────────┼───────────────────────┐                   │
│         │                       │                       │                   │
│         ▼                       ▼                       ▼                   │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐            │
│  │  AudioInput  │       │ StateManager │       │  AudioOutput │            │
│  │  (pyaudio)   │       │              │       │  (say cmd)   │            │
│  └──────┬───────┘       └──────┬───────┘       └──────▲───────┘            │
│         │                      │                      │                     │
│         ▼                      │                      │                     │
│  ┌──────────────┐              │               ┌──────────────┐            │
│  │  WakeWord    │              │               │     TTS      │            │
│  │ (Porcupine)  │              │               │   Manager    │            │
│  └──────┬───────┘              │               └──────▲───────┘            │
│         │                      │                      │                     │
│         ▼                      │                      │                     │
│  ┌──────────────┐              │               ┌──────────────┐            │
│  │     STT      │              │               │   Response   │            │
│  │ (Whisper.cpp)│              │               │   Parser     │            │
│  └──────┬───────┘              │               └──────▲───────┘            │
│         │                      │                      │                     │
│         └──────────────────────┼──────────────────────┘                     │
│                                │                                            │
│                                ▼                                            │
│                        ┌──────────────┐                                     │
│                        │ Claude Code  │                                     │
│                        │     CLI      │                                     │
│                        │ (subprocess) │                                     │
│                        └──────┬───────┘                                     │
│                                │                                            │
│                                ▼                                            │
│                        ┌──────────────┐                                     │
│                        │ MCP Servers  │                                     │
│                        │ GitHub/Slack │                                     │
│                        └──────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
va/
├── README.md
├── requirements.txt
├── jarvis/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── input.py         # Microphone capture
│   │   ├── output.py        # TTS (say command)
│   │   └── vad.py           # Voice activity detection
│   ├── speech/
│   │   ├── __init__.py
│   │   ├── wake_word.py     # Picovoice Porcupine
│   │   └── stt.py           # Whisper.cpp transcription
│   ├── llm/
│   │   ├── __init__.py
│   │   └── claude.py        # Claude Code CLI wrapper
│   └── core/
│       ├── __init__.py
│       ├── state.py         # State machine
│       └── config.py        # Configuration
├── models/
│   └── hey_jarvis.ppn       # Picovoice wake word model
└── tests/
    ├── test_audio.py
    ├── test_stt.py
    └── test_claude.py
```

---

## Dependencies

```txt
# requirements.txt

# Audio
pyaudio>=0.2.14
numpy>=1.24.0
sounddevice>=0.4.6

# Wake Word
pvporcupine>=3.0.0

# Speech-to-Text (choose one)
pywhispercpp>=1.0.0
# OR
openai-whisper>=20231117
# OR use whisper.cpp directly via subprocess

# Async
asyncio-subprocess>=0.0.1

# Utils
python-dotenv>=1.0.0
```

---

## Quick Start Commands

```bash
# 1. Create environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install pyaudio numpy pvporcupine

# 3. Install Whisper.cpp
brew install whisper-cpp
# OR
pip install pywhispercpp

# 4. Get Picovoice key
# Visit: https://console.picovoice.ai/
# Create "Hey Jarvis" model, download .ppn file

# 5. Test components individually
# Test mic
python -c "import pyaudio; print('PyAudio OK')"

# Test TTS
say "Hello, I am Jarvis"

# Test Claude
echo "Say hello" | claude --print --output-format json

# 6. Run POC
python jarvis/main.py
```

---

## Success Criteria for POC

| Criteria | Target | Priority |
|----------|--------|----------|
| Wake word detection | "Hey Jarvis" activates within 500ms | P0 |
| STT accuracy | >95% for clear speech | P0 |
| End-to-end latency | <5 seconds total | P0 |
| Multi-turn conversation | Context maintained for 3+ turns | P0 |
| MCP GitHub query | "What did I commit?" works | P0 |
| MCP Slack post | "Send to Slack" works | P0 |
| Error recovery | Graceful handling, no crashes | P1 |
| Barge-in support | Can interrupt TTS | P2 |

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Whisper.cpp setup issues | Medium | Fallback to OpenAI Whisper API |
| Picovoice accuracy | Low | Tune sensitivity, add confirmation |
| Claude latency spikes | Medium | Cache common responses, use streaming |
| MCP tool failures | Low | Add retry logic, graceful errors |
| Audio permission issues | Medium | Document setup steps clearly |

---

## Next Steps After POC

1. **Better TTS:** Integrate ElevenLabs or OpenAI TTS for natural voice
2. **Menubar App:** Swift wrapper for better macOS integration
3. **Visual Feedback:** Show listening/processing state
4. **Voice Profiles:** Support multiple users
5. **Offline Mode:** Fallback when internet unavailable
6. **Custom Commands:** Direct shortcuts for common tasks

---

## Sources

- [Whisper M4 Benchmarks](https://dev.to/theinsyeds/whisper-speech-recognition-on-mac-m4-performance-analysis-and-benchmarks-2dlp)
- [Picovoice Pricing](https://picovoice.ai/pricing/)
- [Claude Agent SDK](https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk)
- [MCP Protocol Docs](https://modelcontextprotocol.info/docs/)
- [Voice Agent Barge-in](https://sparkco.ai/blog/optimizing-voice-agent-barge-in-detection-for-2025)
- [ElevenLabs vs OpenAI TTS](https://vapi.ai/blog/elevenlabs-vs-openai)

---

*Document Version: 1.0*
*Created: 2026-02-03*
*Target: 10-day POC*
