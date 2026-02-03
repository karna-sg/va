"""
Speech-to-Text Module for Jarvis

Handles:
- Audio transcription using Whisper.cpp
- Model loading and management
- Support for different model sizes
"""

import asyncio
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Try to import pywhispercpp
# Note: pywhispercpp requires Python 3.10+ due to union type syntax
try:
    from pywhispercpp.model import Model as WhisperModel
    PYWHISPERCPP_AVAILABLE = True
except (ImportError, TypeError):
    # TypeError occurs on Python 3.9 due to `bool | TextIO` union syntax
    PYWHISPERCPP_AVAILABLE = False
    WhisperModel = None


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription"""
    text: str
    duration_ms: float
    language: Optional[str] = None
    confidence: Optional[float] = None


class SpeechToText:
    """
    Speech-to-Text using Whisper.cpp

    Supports multiple backends:
    1. pywhispercpp (Python bindings)
    2. whisper.cpp CLI (fallback)
    """

    # Model sizes and their characteristics
    MODELS = {
        'tiny': {'size': '75MB', 'speed': 'fastest', 'accuracy': 'good'},
        'tiny.en': {'size': '75MB', 'speed': 'fastest', 'accuracy': 'good', 'english_only': True},
        'base': {'size': '142MB', 'speed': 'fast', 'accuracy': 'better'},
        'base.en': {'size': '142MB', 'speed': 'fast', 'accuracy': 'better', 'english_only': True},
        'small': {'size': '466MB', 'speed': 'medium', 'accuracy': 'good'},
        'small.en': {'size': '466MB', 'speed': 'medium', 'accuracy': 'good', 'english_only': True},
        'medium': {'size': '1.5GB', 'speed': 'slow', 'accuracy': 'great'},
        'medium.en': {'size': '1.5GB', 'speed': 'slow', 'accuracy': 'great', 'english_only': True},
        'large': {'size': '2.9GB', 'speed': 'slowest', 'accuracy': 'best'},
    }

    # Default model for POC (good balance of speed and accuracy)
    DEFAULT_MODEL = 'base.en'

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        models_dir: Optional[str] = None,
        language: str = 'en',
        use_cli_fallback: bool = True,
    ):
        """
        Initialize Speech-to-Text.

        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            models_dir: Directory containing model files
            language: Language code for transcription
            use_cli_fallback: Use whisper.cpp CLI if pywhispercpp fails
        """
        self.model_name = model_name
        self.language = language
        self.use_cli_fallback = use_cli_fallback

        # Determine models directory
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            # Default to ~/.cache/whisper or project models dir
            self.models_dir = Path.home() / '.cache' / 'whisper'

        self._model: Optional[WhisperModel] = None
        self._model_loaded = False

        # Check for whisper.cpp CLI
        self._cli_available = self._check_cli_available()

    # CLI command names to check (in order of preference)
    CLI_COMMANDS = ['whisper-cli', 'whisper-cpp', 'whisper']

    def _check_cli_available(self) -> bool:
        """Check if whisper.cpp CLI is available"""
        try:
            for cmd in self.CLI_COMMANDS:
                result = subprocess.run(
                    ['which', cmd],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return True
            return False
        except Exception:
            return False

    def _get_cli_command(self) -> str:
        """Get the available CLI command name"""
        for cmd in self.CLI_COMMANDS:
            result = subprocess.run(
                ['which', cmd],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return cmd
        return 'whisper-cli'  # Default fallback

    def _get_model_path(self) -> Path:
        """Get path to model file"""
        # pywhispercpp uses ggml format
        model_file = f"ggml-{self.model_name}.bin"
        return self.models_dir / model_file

    def load_model(self) -> bool:
        """
        Load the Whisper model.

        Returns:
            True if model loaded successfully
        """
        if self._model_loaded:
            return True

        if PYWHISPERCPP_AVAILABLE:
            try:
                print(f"Loading Whisper model: {self.model_name}...")
                start_time = time.time()

                # pywhispercpp can download models automatically
                self._model = WhisperModel(
                    self.model_name,
                    models_dir=str(self.models_dir),
                    print_progress=True
                )

                load_time = time.time() - start_time
                print(f"Model loaded in {load_time:.2f}s")
                self._model_loaded = True
                return True

            except Exception as e:
                print(f"Failed to load model with pywhispercpp: {e}")
                if self.use_cli_fallback and self._cli_available:
                    print("Will use CLI fallback for transcription")
                    self._model_loaded = True
                    return True
                return False

        elif self.use_cli_fallback and self._cli_available:
            print("pywhispercpp not available, using CLI fallback")
            self._model_loaded = True
            return True

        else:
            print("No transcription backend available!")
            print("Install pywhispercpp: pip install pywhispercpp")
            print("Or install whisper.cpp: brew install whisper-cpp")
            return False

    async def transcribe(
        self,
        audio_path: str,
        prompt: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (WAV format, 16kHz mono)
            prompt: Optional prompt to guide transcription

        Returns:
            TranscriptionResult with transcribed text
        """
        if not self._model_loaded:
            if not self.load_model():
                return TranscriptionResult(
                    text="[Error: Model not loaded]",
                    duration_ms=0
                )

        start_time = time.time()

        # Try pywhispercpp first
        if PYWHISPERCPP_AVAILABLE and self._model is not None:
            try:
                result = await self._transcribe_with_pywhispercpp(audio_path)
                result.duration_ms = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                print(f"pywhispercpp transcription failed: {e}")
                if not self.use_cli_fallback:
                    raise

        # Fallback to CLI
        if self._cli_available:
            result = await self._transcribe_with_cli(audio_path)
            result.duration_ms = (time.time() - start_time) * 1000
            return result

        return TranscriptionResult(
            text="[Error: No transcription backend available]",
            duration_ms=0
        )

    async def _transcribe_with_pywhispercpp(
        self,
        audio_path: str
    ) -> TranscriptionResult:
        """Transcribe using pywhispercpp"""
        loop = asyncio.get_event_loop()

        def _transcribe():
            segments = self._model.transcribe(audio_path)
            text = " ".join([seg.text.strip() for seg in segments])
            return text

        text = await loop.run_in_executor(None, _transcribe)

        return TranscriptionResult(
            text=text.strip(),
            duration_ms=0,
            language=self.language
        )

    async def _transcribe_with_cli(self, audio_path: str) -> TranscriptionResult:
        """Transcribe using whisper.cpp CLI"""
        # Get CLI command
        cli_cmd = self._get_cli_command()

        # Build command for whisper-cli (Homebrew version)
        # Format: whisper-cli [options] file
        cmd = [
            cli_cmd,
            '-m', str(self._get_model_path()),
            '-l', self.language,
            '-nt',  # No timestamps
            '-t', '4',  # Number of threads
            '-np',  # No prints except results
            audio_path,  # File goes at the end
        ]

        # Run transcription
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            return TranscriptionResult(
                text=f"[Error: {error_msg}]",
                duration_ms=0
            )

        text = stdout.decode().strip()

        return TranscriptionResult(
            text=text,
            duration_ms=0,
            language=self.language
        )

    async def transcribe_bytes(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transcribe audio from bytes.

        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Sample rate of audio

        Returns:
            TranscriptionResult with transcribed text
        """
        import wave

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            filepath = f.name

        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        try:
            result = await self.transcribe(filepath)
            return result
        finally:
            # Clean up temp file
            Path(filepath).unlink(missing_ok=True)

    def get_model_info(self) -> dict:
        """Get information about current model"""
        info = self.MODELS.get(self.model_name, {})
        return {
            'name': self.model_name,
            'loaded': self._model_loaded,
            'backend': 'pywhispercpp' if (PYWHISPERCPP_AVAILABLE and self._model) else 'cli',
            **info
        }

    @classmethod
    def list_models(cls) -> list[dict]:
        """List available models with their characteristics"""
        return [
            {'name': name, **info}
            for name, info in cls.MODELS.items()
        ]


# Test function
async def test_stt():
    """Test Speech-to-Text functionality"""
    print("Testing Speech-to-Text...")

    # List available models
    print("\nAvailable models:")
    for model in SpeechToText.list_models():
        print(f"  - {model['name']}: {model['size']}, {model['speed']}, {model['accuracy']}")

    # Initialize STT
    stt = SpeechToText(model_name='base.en')

    # Show model info
    print(f"\nModel info: {stt.get_model_info()}")

    # Load model
    print("\nLoading model...")
    if stt.load_model():
        print("Model loaded successfully!")
    else:
        print("Failed to load model")
        return

    # Test with a simple audio file (you would need an actual file)
    print("\nTo test transcription, create a test.wav file and run:")
    print("  result = await stt.transcribe('test.wav')")
    print("  print(result.text)")

    print("\nSTT test complete!")


if __name__ == "__main__":
    asyncio.run(test_stt())
