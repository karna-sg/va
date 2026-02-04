"""
Deepgram Streaming STT Module for Jarvis

Features:
- WebSocket streaming for real-time transcription (<300ms)
- Transcribes as you speak (not after)
- Supports interim results for instant feedback
- Automatic endpointing (detects end of speech)
"""

import asyncio
import os
import json
from typing import Optional, Callable, AsyncGenerator
from dataclasses import dataclass
import base64

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

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class TranscriptionResult:
    """Result from Deepgram transcription"""
    text: str
    is_final: bool  # Segment is finalized (won't change)
    speech_final: bool = False  # Entire speech is done (endpoint detected)
    confidence: float = 0.0
    words: list = None
    duration: float = 0.0

    def __post_init__(self):
        if self.words is None:
            self.words = []


class DeepgramSTT:
    """
    Deepgram streaming speech-to-text.

    Usage:
        stt = DeepgramSTT(api_key="your-key")

        # Simple transcription
        text = await stt.transcribe()

        # Or with callback for real-time updates
        async for result in stt.transcribe_stream():
            print(f"{'[final]' if result.is_final else '[interim]'} {result.text}")
    """

    WEBSOCKET_URL = "wss://api.deepgram.com/v1/listen"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nova-2",
        language: str = "en-US",
        sample_rate: int = 16000,
        channels: int = 1,
        encoding: str = "linear16",
        interim_results: bool = True,
        endpointing: int = 500,  # ms of silence to detect end of speech
        vad_events: bool = True,
    ):
        """
        Initialize Deepgram STT.

        Args:
            api_key: Deepgram API key (or set DEEPGRAM_API_KEY env var)
            model: Model to use (nova-2 is fastest and most accurate)
            language: Language code
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            encoding: Audio encoding (linear16 for PCM)
            interim_results: Return partial transcriptions
            endpointing: Silence duration (ms) to detect end of speech
            vad_events: Enable voice activity detection events
        """
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("Deepgram API key required. Set DEEPGRAM_API_KEY env var.")

        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.channels = channels
        self.encoding = encoding
        self.interim_results = interim_results
        self.endpointing = endpointing
        self.vad_events = vad_events

        # Audio recording settings
        self.chunk_size = 1024
        self.format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None

        # State
        self._is_listening = False
        self._should_stop = False
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

    def _build_url(self) -> str:
        """Build WebSocket URL with parameters"""
        params = [
            f"model={self.model}",
            f"language={self.language}",
            f"sample_rate={self.sample_rate}",
            f"channels={self.channels}",
            f"encoding={self.encoding}",
            f"interim_results={'true' if self.interim_results else 'false'}",
            f"endpointing={self.endpointing}",
            f"vad_events={'true' if self.vad_events else 'false'}",
            "punctuate=true",
            "smart_format=true",
        ]
        return f"{self.WEBSOCKET_URL}?{'&'.join(params)}"

    def _init_audio(self):
        """Initialize PyAudio"""
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not installed. Run: pip install pyaudio")

        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

    def _open_stream(self):
        """Open audio input stream"""
        self._init_audio()

        if self._stream is None or not self._stream.is_active():
            self._stream = self._pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

    def _close_stream(self):
        """Close audio input stream"""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    async def transcribe(
        self,
        timeout: float = 30.0,
        silence_timeout: float = 2.0,
        on_interim: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Record and transcribe speech.

        Args:
            timeout: Maximum recording time in seconds
            silence_timeout: Stop after this much silence (uses Deepgram endpointing)
            on_interim: Callback for interim results

        Returns:
            Final transcribed text
        """
        final_text = ""

        async for result in self.transcribe_stream(timeout=timeout):
            if on_interim and not result.is_final:
                on_interim(result.text)

            if result.is_final and result.text.strip():
                final_text = result.text.strip()
                break

        return final_text

    async def transcribe_stream(
        self,
        timeout: float = 30.0,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Stream transcription results as they arrive.

        Args:
            timeout: Maximum recording time in seconds

        Yields:
            TranscriptionResult objects (interim and final)
        """
        # Lazy import websockets
        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets not installed. Run: pip install websockets")

        self._is_listening = True
        self._should_stop = False

        try:
            self._open_stream()

            url = self._build_url()
            headers = {"Authorization": f"Token {self.api_key}"}

            async with websockets.connect(url, additional_headers=headers) as ws:
                # Task to send audio
                async def send_audio():
                    try:
                        while not self._should_stop:
                            # Read audio chunk
                            data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                            await ws.send(data)
                            await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                    except Exception:
                        pass  # Normal closure, ignore
                    finally:
                        # Send close message
                        try:
                            await ws.send(json.dumps({"type": "CloseStream"}))
                        except:
                            pass

                # Start sending audio in background
                send_task = asyncio.create_task(send_audio())

                # Set timeout
                start_time = asyncio.get_event_loop().time()

                # Receive transcriptions
                try:
                    async for message in ws:
                        if self._should_stop:
                            break

                        # Check timeout
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed > timeout:
                            break

                        try:
                            data = json.loads(message)
                            msg_type = data.get("type", "")

                            if msg_type == "Results":
                                channel = data.get("channel", {})
                                alternatives = channel.get("alternatives", [])

                                if alternatives:
                                    alt = alternatives[0]
                                    text = alt.get("transcript", "")
                                    confidence = alt.get("confidence", 0.0)
                                    words = alt.get("words", [])

                                    is_final = data.get("is_final", False)
                                    speech_final = data.get("speech_final", False)

                                    if text.strip():
                                        result = TranscriptionResult(
                                            text=text,
                                            is_final=is_final,
                                            speech_final=speech_final,
                                            confidence=confidence,
                                            words=words,
                                        )
                                        yield result

                                        # If speech is final (endpoint detected), stop
                                        if speech_final:
                                            self._should_stop = True
                                            break

                            elif msg_type == "SpeechStarted":
                                # Voice activity detected
                                pass

                            elif msg_type == "UtteranceEnd":
                                # End of utterance detected
                                self._should_stop = True
                                break

                        except json.JSONDecodeError:
                            continue

                except websockets.exceptions.ConnectionClosed:
                    pass

                # Cancel send task
                self._should_stop = True
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

        finally:
            self._is_listening = False
            self._close_stream()

    async def stop(self):
        """Stop current transcription"""
        self._should_stop = True

    @property
    def is_listening(self) -> bool:
        """Check if currently listening"""
        return self._is_listening

    def cleanup(self):
        """Clean up resources"""
        self._close_stream()
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None


class DeepgramSTTFallback:
    """
    Fallback STT using local Whisper when Deepgram unavailable.
    """

    def __init__(self, model_name: str = "tiny.en"):
        from jarvis.speech.stt import SpeechToText
        self._stt = SpeechToText(model_name=model_name)
        self._stt.load_model()

    async def transcribe(
        self,
        timeout: float = 30.0,
        silence_timeout: float = 2.0,
        on_interim: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Transcribe using Whisper (non-streaming)"""
        from jarvis.audio.input import AudioInput

        audio_input = AudioInput(sample_rate=16000)
        audio_data = await audio_input.record_until_silence(
            silence_threshold=500,
            silence_duration=silence_timeout,
            max_duration=timeout,
        )
        audio_input.cleanup()

        if len(audio_data) < 1000:
            return ""

        result = await self._stt.transcribe_bytes(audio_data, sample_rate=16000)
        return result.text.strip()

    async def stop(self):
        pass

    @property
    def is_listening(self) -> bool:
        return False

    def cleanup(self):
        pass


def get_stt(prefer_deepgram: bool = True):
    """
    Get the best available STT engine.

    Args:
        prefer_deepgram: If True, try Deepgram first

    Returns:
        STT instance (Deepgram or Whisper fallback)
    """
    if prefer_deepgram and os.getenv("DEEPGRAM_API_KEY"):
        try:
            return DeepgramSTT()
        except Exception as e:
            print(f"Deepgram unavailable ({e}), using Whisper fallback")

    return DeepgramSTTFallback()


# Test
async def test_deepgram():
    """Test Deepgram STT"""
    api_key = os.getenv("DEEPGRAM_API_KEY")

    if not api_key:
        print("No DEEPGRAM_API_KEY set, testing with Whisper fallback...")
        print("(Set DEEPGRAM_API_KEY to test Deepgram streaming)")
        return

    print("Testing Deepgram Streaming STT...")
    print("Speak something (will record for up to 10 seconds)...\n")

    stt = DeepgramSTT(api_key=api_key)

    # Test with interim results
    final_text = ""
    async for result in stt.transcribe_stream(timeout=10.0):
        if result.is_final:
            print(f"\n[FINAL] {result.text}")
            final_text = result.text
            break
        else:
            print(f"[interim] {result.text}", end="\r")

    print(f"\nFinal transcription: {final_text}")
    stt.cleanup()
    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(test_deepgram())
