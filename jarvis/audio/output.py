"""
Audio Output Module - Text-to-Speech for Jarvis

Handles:
- Text-to-speech using macOS `say` command
- Voice selection and configuration
- Interruptible speech playback
- Audio feedback sounds
"""

import asyncio
import subprocess
from typing import Optional
from pathlib import Path


class AudioOutput:
    """Handles text-to-speech and audio output"""

    # Available macOS voices (subset of commonly used ones)
    VOICES = {
        'samantha': 'Samantha',      # Default US English female
        'alex': 'Alex',              # US English male
        'victoria': 'Victoria',       # US English female
        'daniel': 'Daniel',          # British English male
        'karen': 'Karen',            # Australian English female
        'moira': 'Moira',            # Irish English female
        'tessa': 'Tessa',            # South African English female
        'zoe': 'Zoe (Premium)',      # Premium US English female (if installed)
    }

    # System sounds for feedback
    SOUNDS = {
        'activation': '/System/Library/Sounds/Pop.aiff',
        'success': '/System/Library/Sounds/Glass.aiff',
        'error': '/System/Library/Sounds/Basso.aiff',
        'listening': '/System/Library/Sounds/Tink.aiff',
        'done': '/System/Library/Sounds/Ping.aiff',
    }

    def __init__(
        self,
        voice: str = 'samantha',
        rate: int = 200,  # Words per minute (default ~175-200)
    ):
        """
        Initialize audio output.

        Args:
            voice: Voice name to use (see VOICES)
            rate: Speech rate in words per minute
        """
        self.voice = self.VOICES.get(voice.lower(), voice)
        self.rate = rate
        self._current_process: Optional[asyncio.subprocess.Process] = None
        self._is_speaking = False

    @classmethod
    def list_voices(cls) -> list[str]:
        """List all available system voices"""
        result = subprocess.run(
            ['say', '-v', '?'],
            capture_output=True,
            text=True
        )
        voices = []
        for line in result.stdout.strip().split('\n'):
            if line:
                # Format: "Voice Name    language_code  # description"
                voice_name = line.split()[0]
                voices.append(voice_name)
        return voices

    async def speak(self, text: str, wait: bool = True, max_length: int = 500) -> bool:
        """
        Speak text using TTS.

        Args:
            text: Text to speak
            wait: If True, wait for speech to complete
            max_length: Maximum characters to speak (truncate long text)

        Returns:
            True if completed, False if interrupted
        """
        if not text or not text.strip():
            return True

        # Clean text for speech
        text = self._clean_text_for_speech(text)

        # Truncate very long text for voice output
        if len(text) > max_length:
            # Find a good breaking point (end of sentence)
            truncate_at = text.rfind('. ', 0, max_length)
            if truncate_at == -1:
                truncate_at = max_length
            text = text[:truncate_at + 1]

        self._is_speaking = True
        try:
            cmd = ['say', '-v', self.voice, '-r', str(self.rate), text]
            self._current_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )

            if wait:
                await self._current_process.wait()
                return self._current_process.returncode == 0
            return True

        finally:
            if wait:
                self._is_speaking = False
                self._current_process = None

    async def speak_streamed(
        self,
        text_generator,
        sentence_pause: float = 0.1
    ) -> bool:
        """
        Speak text as it streams in, sentence by sentence.

        Args:
            text_generator: Async generator yielding text chunks
            sentence_pause: Pause between sentences

        Returns:
            True if completed, False if interrupted
        """
        buffer = ""
        sentence_endings = {'.', '!', '?', '\n'}

        async for chunk in text_generator:
            buffer += chunk

            # Check for complete sentences
            while any(end in buffer for end in sentence_endings):
                # Find the first sentence ending
                earliest_pos = len(buffer)
                for end in sentence_endings:
                    pos = buffer.find(end)
                    if pos != -1 and pos < earliest_pos:
                        earliest_pos = pos

                # Extract and speak the sentence
                sentence = buffer[:earliest_pos + 1].strip()
                buffer = buffer[earliest_pos + 1:]

                if sentence:
                    success = await self.speak(sentence)
                    if not success:
                        return False
                    await asyncio.sleep(sentence_pause)

        # Speak any remaining text
        if buffer.strip():
            return await self.speak(buffer.strip())

        return True

    async def interrupt(self) -> None:
        """Stop current speech immediately"""
        if self._current_process and self._current_process.returncode is None:
            self._current_process.terminate()
            try:
                await asyncio.wait_for(self._current_process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                self._current_process.kill()
        self._is_speaking = False
        self._current_process = None

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self._is_speaking

    async def play_sound(self, sound_name: str) -> None:
        """
        Play a system sound.

        Args:
            sound_name: Name of sound (see SOUNDS) or path to audio file
        """
        sound_path = self.SOUNDS.get(sound_name, sound_name)

        if not Path(sound_path).exists():
            return

        process = await asyncio.create_subprocess_exec(
            'afplay', sound_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.wait()

    async def play_activation_sound(self) -> None:
        """Play sound to indicate activation (wake word detected)"""
        await self.play_sound('activation')

    async def play_listening_sound(self) -> None:
        """Play sound to indicate listening started"""
        await self.play_sound('listening')

    async def play_success_sound(self) -> None:
        """Play sound to indicate successful operation"""
        await self.play_sound('success')

    async def play_error_sound(self) -> None:
        """Play sound to indicate error"""
        await self.play_sound('error')

    async def play_done_sound(self) -> None:
        """Play sound to indicate done/completed"""
        await self.play_sound('done')

    async def say_thinking(self) -> None:
        """Quick audio cue that processing has started"""
        # Just a short spoken cue - faster than playing a sound file
        await self.speak("Checking.", wait=True, max_length=50)

    def set_voice(self, voice: str) -> None:
        """Change the voice"""
        self.voice = self.VOICES.get(voice.lower(), voice)

    def set_rate(self, rate: int) -> None:
        """Change the speech rate"""
        self.rate = max(100, min(400, rate))  # Clamp to reasonable range

    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech output"""
        import re

        # Remove URLs first (before other processing)
        text = re.sub(r'https?://[^\s]+', '', text)

        # Remove file paths (Unix and Windows style)
        text = re.sub(r'[/\\][\w/\\.-]+\.\w+', '', text)  # paths with extensions
        text = re.sub(r'[/\\]Users[/\\][^\s]+', '', text)  # /Users/... paths
        text = re.sub(r'[/\\]home[/\\][^\s]+', '', text)   # /home/... paths

        # Remove code blocks content
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)  # inline code

        # Remove markdown formatting
        text = text.replace('**', '')
        text = text.replace('*', '')

        # Remove markdown headers but keep the text
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

        # Remove markdown links, keep link text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Remove markdown tables completely
        text = re.sub(r'\|[^\n]+\|', '', text)
        text = re.sub(r'^[-|:\s]+$', '', text, flags=re.MULTILINE)

        # Remove bullet points and list markers, keep content
        text = re.sub(r'^[\s]*[-*•]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s*', '', text, flags=re.MULTILINE)

        # Remove common emoji
        text = re.sub(r'[🔴🟡🟢✅❌📝🎤💬🚀📊👥🗺️🧪]', '', text)

        # Remove git commit hashes (7+ hex chars)
        text = re.sub(r'\b[a-f0-9]{7,40}\b', '', text)

        # Replace symbols with spoken equivalents
        text = text.replace('&', ' and ')
        text = text.replace('@', ' at ')
        text = text.replace('%', ' percent ')
        text = text.replace('→', ' to ')
        text = text.replace('--', ', ')
        text = text.replace('_', ' ')

        # Clean up newlines - convert to periods
        text = re.sub(r'\n+', '. ', text)

        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove multiple punctuation
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r',+', ',', text)
        text = re.sub(r'\.\s*\.', '.', text)
        text = re.sub(r':\s*\.', '.', text)

        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)

        return text.strip()


# Quick test function
async def test_audio_output():
    """Test audio output functionality"""
    print("Testing Audio Output...")

    output = AudioOutput(voice='samantha', rate=200)

    # List available voices
    print("\nAvailable voices (sample):")
    voices = output.list_voices()
    for voice in voices[:10]:  # Show first 10
        print(f"  - {voice}")
    print(f"  ... and {len(voices) - 10} more")

    # Test sounds
    print("\nPlaying activation sound...")
    await output.play_activation_sound()
    await asyncio.sleep(0.5)

    # Test TTS
    print("Speaking test message...")
    await output.speak("Hello, I am Jarvis, your voice assistant.")

    # Test interrupt
    print("\nTesting interrupt (will speak then stop)...")
    speak_task = asyncio.create_task(
        output.speak("This is a long message that should be interrupted before it finishes completely.")
    )
    await asyncio.sleep(1.5)
    await output.interrupt()
    print("Speech interrupted!")

    # Test success sound
    print("\nPlaying success sound...")
    await output.play_success_sound()

    print("\nAudio output test complete!")


if __name__ == "__main__":
    asyncio.run(test_audio_output())
