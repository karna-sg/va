"""
Configuration Module for Jarvis

Handles:
- Application configuration
- Environment variable loading
- Default settings
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Try to load dotenv from project root
try:
    from dotenv import load_dotenv
    # Load from project root (jarvis/core/config.py -> ../../.env)
    _project_root = Path(__file__).parent.parent.parent
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        load_dotenv()  # Fall back to cwd
except ImportError:
    pass


@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    device_index: Optional[int] = None

    # Recording settings - OPTIMIZED for low latency
    silence_threshold: float = 400.0  # More sensitive (was 500)
    silence_duration: float = 0.7  # Faster end detection (was 1.0)
    max_recording_duration: float = 30.0  # seconds

    # TTS settings (macOS fallback)
    tts_voice: str = "Samantha"
    tts_rate: int = 220  # Faster speech rate (was 210)

    # ElevenLabs TTS settings (streaming, low latency)
    use_elevenlabs: bool = False  # Disabled - free tier exhausted, using macOS TTS
    elevenlabs_voice: str = "sarah"  # Female, soft friendly voice
    elevenlabs_model: str = "eleven_turbo_v2_5"  # Fast model ~75ms latency


@dataclass
class STTConfig:
    """Speech-to-text configuration"""
    model_name: str = "tiny.en"  # Fallback model if Deepgram unavailable
    models_dir: Optional[str] = None
    language: str = "en"
    use_cli_fallback: bool = True
    # Model options: tiny.en (fastest), base.en (balanced), small.en (accurate)

    # Deepgram streaming STT settings - ENABLED for <300ms latency
    use_deepgram: bool = True  # ENABLED - much faster than Whisper
    deepgram_model: str = "nova-2"  # Fast and accurate
    deepgram_language: str = "en-US"
    deepgram_endpointing: int = 300  # Aggressive: 300ms silence = end of speech


@dataclass
class WakeWordConfig:
    """Wake word detection configuration"""
    enabled: bool = True  # Enabled -- uses VAD + Deepgram burst (no Picovoice needed)
    wake_phrases: list = None  # Phrases to detect (default: hey jarvis, hi jarvis, etc.)
    burst_duration: float = 2.0  # Seconds of audio to buffer before checking
    sensitivity: float = 0.5

    # Legacy Picovoice fields (unused, kept for compatibility)
    access_key: str = ""
    keyword: str = "jarvis"
    model_path: Optional[str] = None

    def __post_init__(self):
        if self.wake_phrases is None:
            self.wake_phrases = [
                "hey jarvis", "hi jarvis", "hello jarvis",
                "okay jarvis", "ok jarvis",
                "jarvis",
            ]


@dataclass
class ClaudeConfig:
    """Claude Code configuration"""
    permission_mode: str = "bypassPermissions"
    timeout: float = 60.0  # Reduced from 300s - most voice queries are quick
    # Working directory MUST be set to a directory with .mcp.json for MCP tools
    working_directory: str = "/Users/karna/curiescious/curiescious"

    # Project directories where agent can make code changes
    project_directories: list = None

    # Default GitHub owner for the user's repositories
    github_owner: str = "karna-sg"

    # Default repositories to check
    default_repos: list = None

    # Model routing settings (Claude Code CLI short names)
    fast_model: str = "haiku"  # For simple queries (faster, cheaper)
    smart_model: str = "sonnet"  # For complex tasks (tools, code)
    use_model_routing: bool = True  # Enable automatic model selection

    # System prompt for voice agent context - CONVERSATIONAL DESIGN
    system_prompt: str = """You are Jarvis, a friendly female voice assistant for a developer named Vasu. You speak out loud - keep it SHORT.

## CRITICAL RULES (MUST FOLLOW)
1. MAX 2 sentences. No exceptions.
2. NEVER list items. Just say totals: "You have 20 issues" NOT "Here are the issues: 1, 2, 3..."
3. NO markdown, bullets, numbers, or formatting - this is SPOKEN not written.
4. Always offer more: "Want details?" or "Should I pick one?"

## RESPONSE FORMAT
WRONG: "Here are your 20 issues: #1 Setup, #2 Library, #3 Notes, #4..."
RIGHT: "You've got 20 open issues. The top priorities are workspace setup and library management. Want me to pick one to start?"

WRONG: "Issue #20 is titled Product Roadmap with the following details..."
RIGHT: "Issue 20 is about the product roadmap. Want me to read the description?"

## ABOUT VASU (your user)
- Developer working on curiescious (karna-sg/curiescious)
- Uses GitHub for issues/PRs, Slack for team chat, Jira for sprint planning
- Prefers concise updates and action-oriented responses
- Address him as Vasu when appropriate

## GITHUB
- Use MCP tools for GitHub queries
- Default repo: karna-sg/curiescious
- "curious" = curiescious

## PERSONALITY
- Warm and casual: "Hey Vasu!", "Sure!", "Got it!", "Alright..."
- You're a helpful friend, not a robot reading documentation"""

    def __post_init__(self):
        if self.project_directories is None:
            self.project_directories = ["/Users/karna/curiescious/curiescious"]
        if self.default_repos is None:
            self.default_repos = ["curiescious"]


@dataclass
class Config:
    """Main configuration class"""

    # Component configs
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)

    # General settings
    debug: bool = False
    log_level: str = "INFO"

    # User personalization
    user_name: str = "Vasu"

    # Conversation settings
    follow_up_timeout: float = 8.0  # Shorter for Siri-like conversational flow
    max_conversation_turns: int = 50  # more turns for implementation tasks

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    models_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "whisper")

    # Smart agent settings
    auto_continue_on_question: bool = True  # Auto-listen when agent asks a question

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        config = cls()

        # Audio settings
        if os.getenv("JARVIS_SAMPLE_RATE"):
            config.audio.sample_rate = int(os.getenv("JARVIS_SAMPLE_RATE"))
        if os.getenv("JARVIS_SILENCE_THRESHOLD"):
            config.audio.silence_threshold = float(os.getenv("JARVIS_SILENCE_THRESHOLD"))
        if os.getenv("JARVIS_TTS_VOICE"):
            config.audio.tts_voice = os.getenv("JARVIS_TTS_VOICE")
        if os.getenv("JARVIS_TTS_RATE"):
            config.audio.tts_rate = int(os.getenv("JARVIS_TTS_RATE"))

        # STT settings
        if os.getenv("JARVIS_STT_MODEL"):
            config.stt.model_name = os.getenv("JARVIS_STT_MODEL")
        if os.getenv("JARVIS_MODELS_DIR"):
            config.stt.models_dir = os.getenv("JARVIS_MODELS_DIR")
            config.models_dir = Path(os.getenv("JARVIS_MODELS_DIR"))

        # Wake word settings
        if os.getenv("PICOVOICE_ACCESS_KEY"):
            config.wake_word.access_key = os.getenv("PICOVOICE_ACCESS_KEY")
            config.wake_word.enabled = True
        if os.getenv("JARVIS_WAKE_WORD"):
            config.wake_word.keyword = os.getenv("JARVIS_WAKE_WORD")

        # Claude settings
        if os.getenv("JARVIS_CLAUDE_TIMEOUT"):
            config.claude.timeout = float(os.getenv("JARVIS_CLAUDE_TIMEOUT"))
        if os.getenv("JARVIS_WORKING_DIR"):
            config.claude.working_directory = os.getenv("JARVIS_WORKING_DIR")

        # General settings
        config.debug = os.getenv("JARVIS_DEBUG", "").lower() in ("true", "1", "yes")
        if os.getenv("JARVIS_LOG_LEVEL"):
            config.log_level = os.getenv("JARVIS_LOG_LEVEL")

        return config

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration"""
        return cls()

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "channels": self.audio.channels,
                "chunk_size": self.audio.chunk_size,
                "silence_threshold": self.audio.silence_threshold,
                "silence_duration": self.audio.silence_duration,
                "tts_voice": self.audio.tts_voice,
                "tts_rate": self.audio.tts_rate,
            },
            "stt": {
                "model_name": self.stt.model_name,
                "language": self.stt.language,
            },
            "wake_word": {
                "enabled": self.wake_word.enabled,
                "keyword": self.wake_word.keyword,
            },
            "claude": {
                "permission_mode": self.claude.permission_mode,
                "timeout": self.claude.timeout,
            },
            "debug": self.debug,
            "log_level": self.log_level,
        }

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Check audio settings
        if self.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            issues.append(f"Unusual sample rate: {self.audio.sample_rate}")

        # Check STT model
        valid_models = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large']
        if self.stt.model_name not in valid_models:
            issues.append(f"Unknown STT model: {self.stt.model_name}")

        # Check wake word settings (now uses VAD + Deepgram, no Picovoice needed)
        if self.wake_word.enabled and not os.getenv("DEEPGRAM_API_KEY"):
            issues.append("Wake word enabled but no DEEPGRAM_API_KEY set")

        return issues


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config


# Test function
def test_config():
    """Test configuration"""
    print("Testing Configuration...")

    # Create default config
    config = Config.default()
    print(f"\nDefault config:")
    print(f"  Audio sample rate: {config.audio.sample_rate}")
    print(f"  STT model: {config.stt.model_name}")
    print(f"  TTS voice: {config.audio.tts_voice}")

    # Validate
    issues = config.validate()
    if issues:
        print(f"\nValidation issues: {issues}")
    else:
        print("\nConfiguration is valid!")

    # Test env loading
    config_from_env = Config.from_env()
    print(f"\nConfig from env:")
    print(f"  Debug mode: {config_from_env.debug}")

    print("\nConfiguration test complete!")


if __name__ == "__main__":
    test_config()
