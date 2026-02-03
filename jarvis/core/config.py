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

# Try to load dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    device_index: Optional[int] = None

    # Recording settings - optimized for faster response
    silence_threshold: float = 500.0
    silence_duration: float = 1.0  # Reduced from 1.5s to 1.0s for faster detection
    max_recording_duration: float = 30.0  # seconds

    # TTS settings (macOS fallback)
    tts_voice: str = "Samantha"
    tts_rate: int = 210  # Slightly faster speech rate

    # ElevenLabs TTS settings (streaming, low latency)
    use_elevenlabs: bool = True  # Use ElevenLabs if API key available
    elevenlabs_voice: str = "george"  # Warm British male voice
    elevenlabs_model: str = "eleven_turbo_v2_5"  # Fast model ~250ms


@dataclass
class STTConfig:
    """Speech-to-text configuration"""
    model_name: str = "tiny.en"  # Changed to tiny.en for faster response
    models_dir: Optional[str] = None
    language: str = "en"
    use_cli_fallback: bool = True
    # Model options: tiny.en (fastest), base.en (balanced), small.en (accurate)


@dataclass
class WakeWordConfig:
    """Wake word detection configuration (Phase 2)"""
    enabled: bool = False  # Disabled for Phase 1
    access_key: str = ""
    keyword: str = "jarvis"
    sensitivity: float = 0.5
    model_path: Optional[str] = None


@dataclass
class ClaudeConfig:
    """Claude Code configuration"""
    permission_mode: str = "bypassPermissions"
    timeout: float = 300.0  # 5 minutes for complex tasks
    # Working directory MUST be set to a directory with .mcp.json for MCP tools
    working_directory: str = "/Users/karna/curiescious/curiescious"

    # Project directories where agent can make code changes
    project_directories: list = None

    # Default GitHub owner for the user's repositories
    github_owner: str = "karna-sg"

    # Default repositories to check
    default_repos: list = None

    # Model routing settings
    fast_model: str = "haiku"  # For simple queries (faster, cheaper)
    smart_model: str = "sonnet"  # For complex tasks (tools, code)
    use_model_routing: bool = True  # Enable automatic model selection

    # System prompt for voice agent context - CONVERSATIONAL DESIGN
    system_prompt: str = """You are Jarvis, a friendly voice assistant. Talk like a helpful friend, not a robot.

## SPEAKING STYLE (CRITICAL)
- MAX 2 sentences per response. Period.
- NEVER list more than 3 items. Say "you have 20 issues" not list them all.
- ALWAYS end with an offer: "Want me to go through them?" or "Need details?"
- Use casual fillers: "So...", "Alright...", "Let's see...", "Okay so..."
- Be warm: "Hey!", "Sure thing!", "Got it!", "No problem!"

## EXAMPLES OF GOOD RESPONSES
User: "What issues do we have?"
BAD: "Here are all 20 issues: 1. Setup... 2. Library... 3. Notes..." (too long!)
GOOD: "You've got 20 open issues. Top ones are setup, library, and notes. Want me to go through them?"

User: "What did we do yesterday?"
BAD: "Here are the commits from yesterday: commit abc123..." (too detailed!)
GOOD: "Looks like you worked on the sidebar and AI chat panel yesterday. Want the full breakdown?"

User: "Check PRs"
GOOD: "No open PRs right now. Want me to check recent merged ones?"

## GITHUB TOOLS
You have MCP tools - USE THEM for any GitHub question:
- Commits, PRs, issues -> use the tools, don't say "I don't have access"
- Default repo: karna-sg/curiescious

## SPEECH ERRORS (interpret intent)
- "git hub" = GitHub, "curious" = curiescious, "jarrus" = Jarvis
- "summit" = commit, "issue eighteen" = issue 18

## NEVER DO
- Never use markdown, code blocks, or bullet points
- Never list more than 3 items
- Never give long explanations
- Never say "I don't have memory" - you have tools!"""

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

    # Conversation settings
    follow_up_timeout: float = 30.0  # longer timeout for complex tasks
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

        # Check wake word settings
        if self.wake_word.enabled and not self.wake_word.access_key:
            issues.append("Wake word enabled but no Picovoice access key")

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
