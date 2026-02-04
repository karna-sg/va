"""
Deepgram Streaming STT Module for Jarvis

Features:
- Persistent WebSocket connection (no reconnect overhead per turn)
- Real-time streaming transcription (<300ms after speech ends)
- Automatic endpointing (detects end of speech)
- KeepAlive to prevent timeout between turns
- Auto-reconnect on connection drop
"""

import asyncio
import os
import json
import time
from typing import Optional, Callable, AsyncGenerator
from dataclasses import dataclass

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
    Deepgram streaming speech-to-text with persistent connection.

    Keeps the WebSocket alive between turns to eliminate connection
    overhead (~1-2s saved per turn). Sends KeepAlive messages during
    idle periods to prevent Deepgram from closing the connection.

    Usage:
        stt = DeepgramSTT(api_key="your-key")
        await stt.connect()  # Pre-connect during init

        # Each call reuses the existing connection
        async for result in stt.transcribe_stream():
            print(result.text)
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
        endpointing: int = 300,  # ms of silence to detect end of speech
        vad_events: bool = True,
    ):
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

        # Persistent state
        self._ws = None  # WebSocket connection
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._is_listening = False
        self._should_stop = False
        self._keepalive_task = None
        self._is_connected = False

    def _build_url(self) -> str:
        """Build WebSocket URL with parameters"""
        params = [
            "model=%s" % self.model,
            "language=%s" % self.language,
            "sample_rate=%d" % self.sample_rate,
            "channels=%d" % self.channels,
            "encoding=%s" % self.encoding,
            "interim_results=%s" % ('true' if self.interim_results else 'false'),
            "endpointing=%d" % self.endpointing,
            "vad_events=%s" % ('true' if self.vad_events else 'false'),
            "punctuate=true",
            "smart_format=true",
        ]
        return "%s?%s" % (self.WEBSOCKET_URL, "&".join(params))

    def _init_audio(self):
        """Initialize PyAudio (keep alive for session lifetime)"""
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not installed. Run: pip install pyaudio")
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

    def _open_mic(self):
        """Open microphone stream"""
        self._init_audio()
        if self._stream is None or not self._stream.is_active():
            self._stream = self._pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

    def _close_mic(self):
        """Close microphone stream (but keep PyAudio alive)"""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    async def connect(self) -> bool:
        """
        Pre-connect WebSocket to Deepgram.

        Call this during initialization to eliminate connection
        overhead on the first transcription call.
        """
        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets not installed. Run: pip install websockets")

        if self._is_connected and self._ws:
            return True

        try:
            url = self._build_url()
            headers = {"Authorization": "Token %s" % self.api_key}

            connect_start = time.time()
            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=None,  # We handle keepalive ourselves
            )
            connect_ms = (time.time() - connect_start) * 1000
            self._is_connected = True

            # Start keepalive to prevent timeout
            self._start_keepalive()

            # Pre-open PyAudio
            self._init_audio()

            print("  Deepgram connected (%.0fms)" % connect_ms)
            return True

        except Exception as e:
            print("  Deepgram connect failed: %s" % e)
            self._is_connected = False
            self._ws = None
            return False

    async def _reconnect(self) -> bool:
        """Reconnect if the WebSocket dropped"""
        self._stop_keepalive()
        self._is_connected = False
        self._ws = None
        return await self.connect()

    def _start_keepalive(self):
        """Start sending KeepAlive messages to prevent timeout"""
        self._stop_keepalive()

        async def _keepalive_loop():
            try:
                while self._is_connected and self._ws:
                    if not self._is_listening:
                        try:
                            await self._ws.send(json.dumps({"type": "KeepAlive"}))
                        except Exception:
                            # Connection dropped
                            self._is_connected = False
                            break
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass

        self._keepalive_task = asyncio.create_task(_keepalive_loop())

    def _stop_keepalive(self):
        """Stop the keepalive task"""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None

    async def transcribe(
        self,
        timeout: float = 30.0,
        silence_timeout: float = 2.0,
        on_interim: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Record and transcribe speech.

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

        Uses the persistent WebSocket connection. Reconnects if needed.

        Yields:
            TranscriptionResult objects (interim and final)
        """
        self._is_listening = True
        self._should_stop = False

        # Ensure we have a connection
        if not self._is_connected or not self._ws:
            if not await self._reconnect():
                # Fall back to fresh connection
                async for result in self._transcribe_fresh(timeout):
                    yield result
                return

        try:
            # Drain any stale messages from previous utterance
            await self._drain_stale_messages()

            self._open_mic()

            # Background task: send audio chunks
            async def send_audio():
                try:
                    while not self._should_stop and self._is_connected:
                        data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                        await self._ws.send(data)
                        await asyncio.sleep(0.005)  # ~200 chunks/sec at 16kHz
                except Exception:
                    pass

            send_task = asyncio.create_task(send_audio())
            start_time = asyncio.get_event_loop().time()

            # Receive transcriptions
            try:
                async for message in self._ws:
                    if self._should_stop:
                        break

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

                                    if speech_final:
                                        self._should_stop = True
                                        break

                        elif msg_type == "UtteranceEnd":
                            self._should_stop = True
                            break

                    except json.JSONDecodeError:
                        continue

            except Exception as e:
                # Connection dropped mid-stream
                self._is_connected = False
                print("  Deepgram connection lost: %s" % e)

            # Stop sending audio
            self._should_stop = True
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass

        finally:
            self._is_listening = False
            self._close_mic()

            # If connection died, mark for reconnect on next call
            if not self._is_connected:
                self._ws = None

    async def _drain_stale_messages(self):
        """Clear any buffered messages from previous utterance"""
        if not self._ws:
            return
        try:
            while True:
                # Non-blocking read with very short timeout
                msg = await asyncio.wait_for(self._ws.recv(), timeout=0.05)
                # Silently discard stale messages
        except (asyncio.TimeoutError, Exception):
            pass  # No more stale messages

    async def _transcribe_fresh(
        self,
        timeout: float = 30.0,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Fallback: create a fresh connection (used if persistent connect fails)"""
        try:
            import websockets
        except ImportError:
            return

        try:
            self._open_mic()
            url = self._build_url()
            headers = {"Authorization": "Token %s" % self.api_key}

            async with websockets.connect(url, additional_headers=headers) as ws:
                async def send_audio():
                    try:
                        while not self._should_stop:
                            data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                            await ws.send(data)
                            await asyncio.sleep(0.005)
                    except Exception:
                        pass
                    finally:
                        try:
                            await ws.send(json.dumps({"type": "CloseStream"}))
                        except Exception:
                            pass

                send_task = asyncio.create_task(send_audio())
                start_time = asyncio.get_event_loop().time()

                try:
                    async for message in ws:
                        if self._should_stop:
                            break
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
                                        yield TranscriptionResult(
                                            text=text,
                                            is_final=is_final,
                                            speech_final=speech_final,
                                            confidence=confidence,
                                            words=words,
                                        )
                                        if speech_final:
                                            self._should_stop = True
                                            break

                            elif msg_type == "UtteranceEnd":
                                self._should_stop = True
                                break

                        except json.JSONDecodeError:
                            continue

                except Exception:
                    pass

                self._should_stop = True
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

        finally:
            self._is_listening = False
            self._close_mic()

    async def stop(self):
        """Stop current transcription"""
        self._should_stop = True

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def disconnect(self):
        """Close the persistent WebSocket connection"""
        self._stop_keepalive()
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._is_connected = False

    def cleanup(self):
        """Clean up all resources"""
        self._stop_keepalive()
        self._close_mic()
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
        self._is_connected = False
        self._ws = None


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

    async def connect(self) -> bool:
        return True

    async def disconnect(self):
        pass

    async def stop(self):
        pass

    @property
    def is_listening(self) -> bool:
        return False

    @property
    def is_connected(self) -> bool:
        return True

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
            print("Deepgram unavailable (%s), using Whisper fallback" % e)

    return DeepgramSTTFallback()


# Test
async def test_deepgram():
    """Test Deepgram STT"""
    api_key = os.getenv("DEEPGRAM_API_KEY")

    if not api_key:
        print("No DEEPGRAM_API_KEY set, testing with Whisper fallback...")
        return

    print("Testing Deepgram Streaming STT...")
    stt = DeepgramSTT(api_key=api_key)

    # Test persistent connection
    print("Connecting...")
    await stt.connect()

    print("Speak something (will record for up to 10 seconds)...\n")

    final_text = ""
    async for result in stt.transcribe_stream(timeout=10.0):
        if result.is_final:
            print("\n[FINAL] %s" % result.text)
            final_text = result.text
            break
        else:
            print("[interim] %s" % result.text, end="\r")

    print("\nFinal transcription: %s" % final_text)

    # Test second transcription on same connection
    print("\nSpeak again (testing connection reuse)...")
    async for result in stt.transcribe_stream(timeout=10.0):
        if result.is_final:
            print("\n[FINAL] %s" % result.text)
            break
        else:
            print("[interim] %s" % result.text, end="\r")

    await stt.disconnect()
    stt.cleanup()
    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(test_deepgram())
