# Jarvis Voice Agent

A voice-controlled AI assistant powered by Claude Code CLI with GitHub MCP integration.

## Features

- **Voice Input:** Speech-to-text using Whisper
- **AI Backend:** Claude Code CLI with MCP tools
- **Voice Output:** Text-to-speech with streaming support
- **GitHub Integration:** Check commits, PRs, issues via MCP
- **Conversation Memory:** Multi-turn conversations with session persistence

## Architecture

```
Audio Input → STT (Whisper) → Claude Code + MCP → TTS → Audio Output
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the voice agent
python -m jarvis.main

# Or run in debug mode
python -m jarvis.main --debug
```

## Usage

1. Press Enter to start listening
2. Speak your command
3. Jarvis responds via voice
4. Press Enter for follow-up or wait for timeout

## Example Commands

- "What did we do yesterday?" - Check recent commits
- "Show me open issues" - List GitHub issues
- "Check pull requests" - List open PRs
- "Implement issue 5" - Start implementing a GitHub issue

## Project Structure

```
jarvis/
├── main.py              # Main entry point
├── audio/
│   ├── input.py         # Audio recording
│   ├── output.py        # TTS output
│   └── vad.py           # Voice activity detection
├── speech/
│   ├── stt.py           # Speech-to-text (Whisper)
│   └── corrections.py   # Transcription corrections
├── llm/
│   └── claude.py        # Claude Code CLI integration
└── core/
    ├── config.py        # Configuration
    ├── state.py         # State machine
    └── speech_buffer.py # Smart TTS buffering
```

## Configuration

Environment variables:
- `JARVIS_DEBUG` - Enable debug mode
- `JARVIS_STT_MODEL` - Whisper model (default: tiny.en)
- `JARVIS_TTS_VOICE` - macOS voice (default: Samantha)
- `JARVIS_TTS_RATE` - Speech rate in WPM (default: 210)

## Requirements

- macOS (for TTS via `say` command)
- Python 3.9+
- Claude Code CLI installed and configured
- MCP servers configured for GitHub access

## License

MIT
