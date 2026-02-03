"""
Voice Activity Detection (VAD) Module for Jarvis

Handles:
- Detecting speech vs silence in audio frames
- End-of-speech detection
- Barge-in detection (user speaking during TTS)
"""

import numpy as np
from typing import Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum


class VADState(Enum):
    """Voice Activity Detection states"""
    SILENCE = "silence"
    SPEECH = "speech"
    UNCERTAIN = "uncertain"


@dataclass
class VADResult:
    """Result from VAD processing"""
    state: VADState
    energy: float
    is_speech: bool
    confidence: float


class VoiceActivityDetector:
    """
    Simple energy-based Voice Activity Detection.

    For production, consider using:
    - Silero VAD (more accurate, ML-based)
    - WebRTC VAD
    - py-webrtcvad
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 500.0,
        speech_pad_ms: int = 300,
        min_speech_duration_ms: int = 250,
        max_silence_duration_ms: int = 1500,
    ):
        """
        Initialize VAD.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame duration in milliseconds
            energy_threshold: RMS energy threshold for speech detection
            speech_pad_ms: Padding around speech segments
            min_speech_duration_ms: Minimum duration to consider as speech
            max_silence_duration_ms: Maximum silence before end-of-speech
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        self.speech_pad_ms = speech_pad_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_silence_duration_ms = max_silence_duration_ms

        # Calculate frame size
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        # State tracking
        self._state = VADState.SILENCE
        self._speech_frames = 0
        self._silence_frames = 0

        # Adaptive threshold
        self._energy_history = deque(maxlen=100)
        self._adaptive_threshold: Optional[float] = None

        # Frames needed for state transitions
        self._min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self._max_silence_frames = int(max_silence_duration_ms / frame_duration_ms)
        self._pad_frames = int(speech_pad_ms / frame_duration_ms)

    def process_frame(self, audio_frame: np.ndarray) -> VADResult:
        """
        Process a single audio frame.

        Args:
            audio_frame: Audio samples as numpy array (int16 or float32)

        Returns:
            VADResult with detection state
        """
        # Ensure proper format
        if audio_frame.dtype == np.int16:
            audio_float = audio_frame.astype(np.float32) / 32768.0
        else:
            audio_float = audio_frame.astype(np.float32)

        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(audio_float ** 2)) * 32768  # Scale back for threshold

        # Update energy history for adaptive threshold
        self._energy_history.append(energy)

        # Determine if current frame is speech
        threshold = self._get_threshold()
        is_speech_frame = energy > threshold

        # Update state machine
        confidence = self._update_state(is_speech_frame, energy, threshold)

        return VADResult(
            state=self._state,
            energy=energy,
            is_speech=self._state == VADState.SPEECH,
            confidence=confidence
        )

    def _get_threshold(self) -> float:
        """Get current energy threshold (adaptive or fixed)"""
        if self._adaptive_threshold is not None:
            return self._adaptive_threshold

        if len(self._energy_history) >= 20:
            # Use adaptive threshold based on noise floor
            noise_floor = np.percentile(list(self._energy_history), 10)
            self._adaptive_threshold = max(
                self.energy_threshold,
                noise_floor * 3  # 3x noise floor
            )
            return self._adaptive_threshold

        return self.energy_threshold

    def _update_state(
        self,
        is_speech_frame: bool,
        energy: float,
        threshold: float
    ) -> float:
        """Update VAD state machine"""
        if is_speech_frame:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            # Don't reset speech frames immediately (hangover)
            if self._silence_frames > self._pad_frames:
                self._speech_frames = max(0, self._speech_frames - 1)

        # State transitions
        if self._state == VADState.SILENCE:
            if self._speech_frames >= self._min_speech_frames:
                self._state = VADState.SPEECH
                return 0.8
            elif self._speech_frames > 0:
                self._state = VADState.UNCERTAIN
                return 0.5
            return 0.9

        elif self._state == VADState.UNCERTAIN:
            if self._speech_frames >= self._min_speech_frames:
                self._state = VADState.SPEECH
                return 0.8
            elif self._silence_frames >= self._max_silence_frames:
                self._state = VADState.SILENCE
                self._speech_frames = 0
                return 0.9
            return 0.5

        else:  # SPEECH
            if self._silence_frames >= self._max_silence_frames:
                self._state = VADState.SILENCE
                self._speech_frames = 0
                return 0.9
            return 0.8

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Simple check if frame contains speech"""
        result = self.process_frame(audio_frame)
        return result.is_speech

    def is_end_of_speech(self) -> bool:
        """Check if speech has ended (silence after speech)"""
        return (
            self._state == VADState.SILENCE and
            self._silence_frames >= self._max_silence_frames
        )

    def reset(self) -> None:
        """Reset VAD state"""
        self._state = VADState.SILENCE
        self._speech_frames = 0
        self._silence_frames = 0
        self._adaptive_threshold = None
        self._energy_history.clear()

    def set_threshold(self, threshold: float) -> None:
        """Set energy threshold manually"""
        self.energy_threshold = threshold
        self._adaptive_threshold = None  # Reset adaptive threshold

    def get_energy_stats(self) -> dict:
        """Get energy statistics from history"""
        if not self._energy_history:
            return {'min': 0, 'max': 0, 'mean': 0, 'threshold': self.energy_threshold}

        energies = list(self._energy_history)
        return {
            'min': float(np.min(energies)),
            'max': float(np.max(energies)),
            'mean': float(np.mean(energies)),
            'threshold': self._get_threshold()
        }


class BargeInDetector:
    """
    Detects when user starts speaking during TTS playback.
    Used to interrupt TTS when user wants to speak.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 800.0,  # Higher threshold to avoid TTS echo
        confirmation_frames: int = 5,      # Frames needed to confirm barge-in
    ):
        """
        Initialize barge-in detector.

        Args:
            sample_rate: Audio sample rate
            energy_threshold: Energy threshold (higher to ignore TTS)
            confirmation_frames: Number of consecutive speech frames to confirm
        """
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
            min_speech_duration_ms=150,  # Quick detection
            max_silence_duration_ms=500,
        )
        self.confirmation_frames = confirmation_frames
        self._consecutive_speech = 0

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Process frame and detect barge-in.

        Args:
            audio_frame: Audio samples

        Returns:
            True if barge-in detected
        """
        result = self.vad.process_frame(audio_frame)

        if result.is_speech:
            self._consecutive_speech += 1
            if self._consecutive_speech >= self.confirmation_frames:
                return True
        else:
            self._consecutive_speech = 0

        return False

    def reset(self) -> None:
        """Reset detector state"""
        self.vad.reset()
        self._consecutive_speech = 0


# Test function
def test_vad():
    """Test VAD functionality"""
    import asyncio

    print("Testing Voice Activity Detection...")

    vad = VoiceActivityDetector(
        sample_rate=16000,
        energy_threshold=500
    )

    # Simulate silence
    silence = np.zeros(480, dtype=np.int16)
    result = vad.process_frame(silence)
    print(f"Silence frame: state={result.state.value}, energy={result.energy:.1f}")

    # Simulate speech (random noise with higher amplitude)
    speech = (np.random.randn(480) * 5000).astype(np.int16)
    for _ in range(10):  # Process multiple frames
        result = vad.process_frame(speech)
    print(f"Speech frame: state={result.state.value}, energy={result.energy:.1f}")

    # Back to silence
    for _ in range(50):
        result = vad.process_frame(silence)
    print(f"After silence: state={result.state.value}, end_of_speech={vad.is_end_of_speech()}")

    stats = vad.get_energy_stats()
    print(f"Energy stats: {stats}")

    print("\nVAD test complete!")


if __name__ == "__main__":
    test_vad()
