"""
Audio Input Module - Microphone capture for Jarvis

Handles:
- Microphone access and configuration
- Audio stream capture (16kHz, 16-bit, mono)
- Continuous listening for wake word detection
- Recording user speech after activation
"""

import asyncio
import wave
import tempfile
from typing import Optional, Callable, AsyncGenerator
from pathlib import Path

import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: pyaudio not installed. Run: pip install pyaudio")


class AudioInput:
    """Handles microphone input and audio recording"""

    # Audio configuration for speech recognition
    SAMPLE_RATE = 16000  # 16kHz - standard for speech
    CHANNELS = 1         # Mono
    CHUNK_SIZE = 1024    # Samples per buffer
    FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
    BYTES_PER_SAMPLE = 2  # 16-bit = 2 bytes

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        chunk_size: int = CHUNK_SIZE,
        device_index: Optional[int] = None
    ):
        """
        Initialize audio input.

        Args:
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of channels (default: 1 for mono)
            chunk_size: Samples per buffer (default: 1024)
            device_index: Specific input device index (None for default)
        """
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("pyaudio is required. Install with: pip install pyaudio")

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._is_recording = False

    def _ensure_initialized(self):
        """Initialize PyAudio if not already done"""
        if self._pa is None:
            self._pa = pyaudio.PyAudio()

    def list_devices(self) -> list[dict]:
        """List available audio input devices"""
        self._ensure_initialized()
        devices = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate'])
                })
        return devices

    def get_default_device(self) -> dict:
        """Get default input device info"""
        self._ensure_initialized()
        try:
            info = self._pa.get_default_input_device_info()
            return {
                'index': info['index'],
                'name': info['name'],
                'channels': info['maxInputChannels'],
                'sample_rate': int(info['defaultSampleRate'])
            }
        except IOError:
            raise RuntimeError("No default input device found")

    def start_stream(self) -> None:
        """Start the audio input stream"""
        self._ensure_initialized()
        if self._stream is not None:
            return  # Already started

        self._stream = self._pa.open(
            format=self.FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )

    def stop_stream(self) -> None:
        """Stop the audio input stream"""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def read_chunk(self) -> bytes:
        """Read a single chunk of audio data (blocking)"""
        if self._stream is None:
            self.start_stream()
        return self._stream.read(self.chunk_size, exception_on_overflow=False)

    def read_chunk_as_array(self) -> np.ndarray:
        """Read a chunk and return as numpy array"""
        data = self.read_chunk()
        return np.frombuffer(data, dtype=np.int16)

    async def read_chunk_async(self) -> bytes:
        """Read a chunk asynchronously (non-blocking)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read_chunk)

    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
        """Async generator that yields audio chunks continuously"""
        self.start_stream()
        try:
            while True:
                chunk = await self.read_chunk_async()
                yield chunk
        finally:
            self.stop_stream()

    async def record_until_silence(
        self,
        silence_threshold: float = 500,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
        min_speech_duration: float = 0.3,
        wait_for_speech_timeout: float = 10.0,
        on_audio: Optional[Callable[[np.ndarray], None]] = None
    ) -> bytes:
        """
        Record audio until silence is detected, but only AFTER speech starts.

        This method waits for the user to START speaking, then records until
        they stop (silence detected). This prevents capturing empty audio.

        Args:
            silence_threshold: RMS amplitude below which is considered silence
            silence_duration: Seconds of silence to stop recording (after speech)
            max_duration: Maximum recording duration in seconds
            min_speech_duration: Minimum speech duration before silence detection starts
            wait_for_speech_timeout: Max seconds to wait for user to start speaking
            on_audio: Optional callback for each audio chunk

        Returns:
            Recorded audio as bytes
        """
        self.start_stream()
        self._is_recording = True

        frames = []
        silent_chunks = 0
        speech_chunks = 0
        max_silent_chunks = int(silence_duration * self.sample_rate / self.chunk_size)
        min_speech_chunks = int(min_speech_duration * self.sample_rate / self.chunk_size)
        max_chunks = int(max_duration * self.sample_rate / self.chunk_size)
        wait_chunks = int(wait_for_speech_timeout * self.sample_rate / self.chunk_size)
        chunks_recorded = 0
        speech_started = False

        try:
            # Phase 1: Wait for speech to start
            while self._is_recording and chunks_recorded < wait_chunks:
                data = await self.read_chunk_async()
                chunks_recorded += 1

                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

                if rms >= silence_threshold:
                    # Speech detected! Start recording from here
                    speech_started = True
                    frames.append(data)
                    speech_chunks = 1
                    break

            if not speech_started:
                # No speech detected within timeout
                return b''

            # Phase 2: Record until silence (after minimum speech duration)
            while self._is_recording and chunks_recorded < max_chunks:
                data = await self.read_chunk_async()
                frames.append(data)
                chunks_recorded += 1

                audio_data = np.frombuffer(data, dtype=np.int16)

                if on_audio:
                    on_audio(audio_data)

                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

                if rms >= silence_threshold:
                    # Still speaking
                    speech_chunks += 1
                    silent_chunks = 0
                else:
                    # Silence detected
                    silent_chunks += 1
                    # Only stop if we've had enough speech AND enough silence
                    if speech_chunks >= min_speech_chunks and silent_chunks >= max_silent_chunks:
                        break

        finally:
            self._is_recording = False

        return b''.join(frames)

    def stop_recording(self) -> None:
        """Stop current recording"""
        self._is_recording = False

    async def record_for_duration(self, duration: float) -> bytes:
        """
        Record for a fixed duration.

        Args:
            duration: Recording duration in seconds

        Returns:
            Recorded audio as bytes
        """
        self.start_stream()
        frames = []
        num_chunks = int(duration * self.sample_rate / self.chunk_size)

        for _ in range(num_chunks):
            data = await self.read_chunk_async()
            frames.append(data)

        return b''.join(frames)

    def save_to_wav(self, audio_data: bytes, filepath: str) -> str:
        """
        Save audio data to a WAV file.

        Args:
            audio_data: Raw audio bytes
            filepath: Output file path

        Returns:
            Path to saved file
        """
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.BYTES_PER_SAMPLE)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        return filepath

    def save_to_temp_wav(self, audio_data: bytes) -> str:
        """
        Save audio data to a temporary WAV file.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Path to temporary file
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            filepath = f.name
        return self.save_to_wav(audio_data, filepath)

    def cleanup(self) -> None:
        """Clean up resources"""
        self.stop_stream()
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def __enter__(self):
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    async def __aenter__(self):
        self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Quick test function
async def test_audio_input():
    """Test audio input functionality"""
    print("Testing Audio Input...")

    with AudioInput() as audio:
        # List devices
        print("\nAvailable input devices:")
        for device in audio.list_devices():
            print(f"  [{device['index']}] {device['name']}")

        # Get default device
        default = audio.get_default_device()
        print(f"\nDefault device: {default['name']}")

        # Record for 3 seconds
        print("\nRecording for 3 seconds...")
        audio_data = await audio.record_for_duration(3.0)
        print(f"Recorded {len(audio_data)} bytes")

        # Save to file
        filepath = audio.save_to_temp_wav(audio_data)
        print(f"Saved to: {filepath}")

        # Test silence detection
        print("\nRecording until silence (speak then stop)...")
        audio_data = await audio.record_until_silence(
            silence_threshold=500,
            silence_duration=1.5,
            max_duration=10.0
        )
        print(f"Recorded {len(audio_data)} bytes")

    print("\nAudio input test complete!")


if __name__ == "__main__":
    asyncio.run(test_audio_input())
