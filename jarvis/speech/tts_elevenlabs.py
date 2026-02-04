"""
ElevenLabs Streaming TTS Module for Jarvis

Features:
- WebSocket streaming for ultra-low latency (~75ms)
- Plays audio as it's generated
- Supports interruption
- Automatic voice selection
"""

import asyncio
import os
import json
import base64
import wave
import tempfile
from typing import Optional, AsyncGenerator
from dataclasses import dataclass

# websockets will be imported lazily when needed

# Force IPv4 for WebSocket connections (IPv6 routing issues on some networks)
import socket
_orig_getaddrinfo = socket.getaddrinfo
def _getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _getaddrinfo_ipv4

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


@dataclass
class ElevenLabsConfig:
    """ElevenLabs configuration"""
    api_key: str
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # George - natural male voice
    model_id: str = "eleven_turbo_v2_5"  # Fast model, ~250ms latency
    stability: float = 0.5
    similarity_boost: float = 0.75
    output_format: str = "pcm_22050"  # PCM for streaming playback


# Popular voice IDs
VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",  # Female, calm
    "drew": "29vD33N1CtxCmqQRPOHJ",    # Male, friendly
    "clyde": "2EiwWnXFnvU5JabPnv8n",   # Male, war veteran
    "paul": "5Q0t7uMcjvnagumLfvZi",    # Male, news anchor
    "domi": "AZnzlk1XvdvUeBnXmlld",    # Female, strong
    "george": "JBFqnCBsd6RMkjVDRZzb",  # Male, warm British
    "dave": "CYw3kZ02Hs0563khs1Fj",    # Male, conversational
    "sarah": "EXAVITQu4vr4xnSDxMaL",   # Female, soft
    "adam": "pNInz6obpgDQGcFmaJgB",    # Male, deep
}


class ElevenLabsTTS:
    """
    ElevenLabs streaming TTS with real-time playback.

    Usage:
        tts = ElevenLabsTTS(api_key="your-key")
        await tts.speak("Hello, how are you?")

        # Or stream from a generator:
        async for _ in tts.speak_streaming(text_generator):
            pass
    """

    WEBSOCKET_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "george",
        model_id: str = "eleven_turbo_v2_5",
    ):
        """
        Initialize ElevenLabs TTS.

        Args:
            api_key: ElevenLabs API key (or set ELEVENLABS_API_KEY env var)
            voice_id: Voice name (from VOICES) or raw voice ID
            model_id: Model to use (eleven_turbo_v2_5 for speed)
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key required. Set ELEVENLABS_API_KEY env var.")

        # Resolve voice name to ID
        self.voice_id = VOICES.get(voice_id.lower(), voice_id)
        self.model_id = model_id

        # Audio settings
        self.sample_rate = 22050  # PCM 22050 Hz
        self.channels = 1
        self.sample_width = 2  # 16-bit

        # State
        self._is_speaking = False
        self._should_stop = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()

        # PyAudio for playback
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

    def _init_audio(self):
        """Initialize PyAudio for playback"""
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not installed. Run: pip install pyaudio")

        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

    def _open_stream(self):
        """Open audio output stream"""
        self._init_audio()

        if self._stream is None or not self._stream.is_active():
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=1024,
            )

    def _close_stream(self):
        """Close audio output stream"""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    async def speak(self, text: str) -> bool:
        """
        Speak text using streaming TTS.

        Args:
            text: Text to speak

        Returns:
            True if completed, False if interrupted
        """
        if not text or not text.strip():
            return True

        # Lazy import websockets
        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets not installed. Run: pip install websockets")

        self._is_speaking = True
        self._should_stop = False

        try:
            self._open_stream()

            url = self.WEBSOCKET_URL.format(voice_id=self.voice_id)
            url += f"?model_id={self.model_id}&output_format=pcm_22050"

            async with websockets.connect(
                url,
                additional_headers={"xi-api-key": self.api_key}
            ) as ws:
                # Send initial config
                await ws.send(json.dumps({
                    "text": " ",  # Required initial text
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                    "generation_config": {
                        "chunk_length_schedule": [120, 160, 250, 290]
                    }
                }))

                # Send the actual text
                await ws.send(json.dumps({
                    "text": text,
                }))

                # Signal end of text
                await ws.send(json.dumps({
                    "text": ""
                }))

                # Receive and play audio chunks
                async for message in ws:
                    if self._should_stop:
                        break

                    data = json.loads(message)

                    if "audio" in data and data["audio"]:
                        # Decode base64 audio and play
                        audio_bytes = base64.b64decode(data["audio"])
                        self._stream.write(audio_bytes)

                    if data.get("isFinal"):
                        break

            return not self._should_stop

        except Exception as e:
            # Ignore normal WebSocket close errors
            err_str = str(e).lower()
            if "close frame" not in err_str and "1000" not in err_str:
                print(f"ElevenLabs TTS error: {e}")
            return False

        finally:
            self._is_speaking = False

    async def speak_streaming(
        self,
        text_generator: AsyncGenerator[str, None],
        min_chunk_size: int = 50,
    ) -> bool:
        """
        Stream text to TTS as it arrives.

        This connects to ElevenLabs WebSocket and sends text chunks
        as they arrive from the generator, enabling real-time TTS
        while Claude is still generating.

        Args:
            text_generator: Async generator yielding text chunks
            min_chunk_size: Minimum characters before sending to TTS

        Returns:
            True if completed, False if interrupted
        """
        # Lazy import websockets
        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets not installed. Run: pip install websockets")

        self._is_speaking = True
        self._should_stop = False

        try:
            self._open_stream()

            url = self.WEBSOCKET_URL.format(voice_id=self.voice_id)
            url += f"?model_id={self.model_id}&output_format=pcm_22050"

            async with websockets.connect(
                url,
                additional_headers={"xi-api-key": self.api_key}
            ) as ws:
                # Send initial config
                await ws.send(json.dumps({
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                    "generation_config": {
                        "chunk_length_schedule": [50, 120, 200, 260]  # Smaller chunks for streaming
                    }
                }))

                # Task to receive and play audio
                async def receive_audio():
                    async for message in ws:
                        if self._should_stop:
                            break
                        data = json.loads(message)
                        if "audio" in data and data["audio"]:
                            audio_bytes = base64.b64decode(data["audio"])
                            self._stream.write(audio_bytes)
                        if data.get("isFinal"):
                            break

                # Task to send text chunks
                async def send_text():
                    buffer = ""
                    async for chunk in text_generator:
                        if self._should_stop:
                            break
                        buffer += chunk

                        # Send when we have enough text or hit sentence boundary
                        if len(buffer) >= min_chunk_size or any(p in buffer for p in '.!?'):
                            await ws.send(json.dumps({"text": buffer}))
                            buffer = ""

                    # Send remaining buffer
                    if buffer:
                        await ws.send(json.dumps({"text": buffer}))

                    # Signal end of text
                    await ws.send(json.dumps({"text": ""}))

                # Run both tasks concurrently
                await asyncio.gather(send_text(), receive_audio())

            return not self._should_stop

        except Exception as e:
            print(f"ElevenLabs streaming TTS error: {e}")
            return False

        finally:
            self._is_speaking = False

    async def interrupt(self):
        """Stop current speech"""
        self._should_stop = True
        self._close_stream()

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self._is_speaking

    def cleanup(self):
        """Clean up resources"""
        self._close_stream()
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None


class ElevenLabsTTSFallback:
    """
    Fallback TTS that uses macOS say command if ElevenLabs unavailable.
    """

    def __init__(self, voice: str = "Samantha", rate: int = 200):
        self.voice = voice
        self.rate = rate
        self._process: Optional[asyncio.subprocess.Process] = None
        self._is_speaking = False

    async def speak(self, text: str) -> bool:
        """Speak using macOS say command"""
        if not text or not text.strip():
            return True

        self._is_speaking = True
        try:
            self._process = await asyncio.create_subprocess_exec(
                'say', '-v', self.voice, '-r', str(self.rate), text,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await self._process.wait()
            return self._process.returncode == 0
        finally:
            self._is_speaking = False
            self._process = None

    async def interrupt(self):
        """Stop speaking"""
        if self._process and self._process.returncode is None:
            self._process.terminate()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def cleanup(self):
        pass


from typing import Union

def get_tts(prefer_elevenlabs: bool = True) -> Union[ElevenLabsTTS, ElevenLabsTTSFallback]:
    """
    Get the best available TTS engine.

    Args:
        prefer_elevenlabs: If True, try ElevenLabs first

    Returns:
        TTS instance (ElevenLabs or macOS fallback)
    """
    if prefer_elevenlabs and os.getenv("ELEVENLABS_API_KEY"):
        try:
            return ElevenLabsTTS()
        except Exception as e:
            print(f"ElevenLabs unavailable ({e}), using macOS TTS")

    return ElevenLabsTTSFallback()


# Test
async def test_elevenlabs():
    """Test ElevenLabs TTS"""
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        print("No ELEVENLABS_API_KEY set, testing fallback...")
        tts = ElevenLabsTTSFallback()
        await tts.speak("Hello, I am Jarvis using the fallback voice.")
        return

    print("Testing ElevenLabs TTS...")
    tts = ElevenLabsTTS(api_key=api_key, voice_id="george")

    # Test simple speech
    print("Speaking: 'Hello, I am Jarvis.'")
    await tts.speak("Hello, I am Jarvis, your voice assistant.")

    # Test streaming
    print("\nTesting streaming...")

    async def text_generator():
        sentences = [
            "Let me check ",
            "your GitHub ",
            "repository. ",
            "You have 20 open issues. ",
            "Want me to go through them?",
        ]
        for s in sentences:
            yield s
            await asyncio.sleep(0.1)  # Simulate streaming delay

    await tts.speak_streaming(text_generator())

    tts.cleanup()
    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(test_elevenlabs())
