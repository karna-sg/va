"""
Wake Word Listener for Jarvis

Detects "Hey Jarvis" (and variants) using energy-based speech detection + Deepgram REST burst.

How it works:
1. Continuously read mic chunks and compute RMS energy
2. When energy rises above the noise floor (adaptive), start buffering audio
3. Buffer until speech ends (energy drops) or burst_duration reached
4. Send buffered audio to Deepgram REST API for transcription
5. Check if transcription starts with a wake phrase
6. If yes, return WakeWordResult with any trailing text
7. If no, reset and go back to listening

Uses simple energy threshold (not full VAD state machine) because:
- Wake word detection needs to be very sensitive (catch all speech)
- False positives are cheap (just a wasted ~200ms Deepgram REST call)
- The Deepgram transcription check filters out non-wake-word speech
"""

import asyncio
import re
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass


WAKE_PHRASES = [
    "hey jarvis",
    "hi jarvis",
    "hello jarvis",
    "okay jarvis",
    "ok jarvis",
    "jarvis",
]


@dataclass
class WakeWordResult:
    """Result from wake word detection"""
    phrase: str           # The wake phrase detected (e.g. "hey jarvis")
    trailing_text: str    # Text after the wake phrase (e.g. "show me issues")
    confidence: float     # Transcription confidence
    detection_time_ms: float  # How long detection took


class WakeWordListener:
    """
    Listens for "Hey Jarvis" wake word using energy detection + Deepgram REST burst.

    Usage:
        listener = WakeWordListener(audio_input, vad, deepgram_stt)
        result = await listener.listen_for_wake_word()
        print(f"Detected: {result.phrase}, trailing: {result.trailing_text}")
    """

    def __init__(
        self,
        audio_input,
        vad,
        deepgram_stt,
        burst_duration: float = 2.5,
        wake_phrases: list = None,
        debug: bool = False,
    ):
        """
        Args:
            audio_input: AudioInput instance for mic access
            vad: VoiceActivityDetector (used for end-of-speech, not onset)
            deepgram_stt: DeepgramSTT instance (uses transcribe_audio REST method)
            burst_duration: Max seconds to buffer before transcribing
            wake_phrases: List of wake phrases to detect
            debug: Print energy levels for debugging
        """
        self.audio_input = audio_input
        self.vad = vad
        self.stt = deepgram_stt
        self.burst_duration = burst_duration
        self.wake_phrases = wake_phrases or WAKE_PHRASES
        self.debug = debug
        self._should_stop = False

        # Adaptive noise floor tracking
        self._noise_floor = 0.0
        self._noise_samples = []
        self._noise_calibrated = False
        self._CALIBRATION_FRAMES = 8  # ~500ms of audio for calibration (was 15)
        self._SPEECH_MULTIPLIER = 3.0  # Speech must be 3x noise floor
        self._MIN_THRESHOLD = 30.0  # Absolute minimum (catches even quiet mics)

    def _compute_rms(self, audio_bytes: bytes) -> float:
        """Compute RMS energy from raw audio bytes"""
        array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if len(array) == 0:
            return 0.0
        return float(np.sqrt(np.mean(array ** 2)))

    def _is_speech_energy(self, rms: float) -> bool:
        """Check if RMS energy indicates speech (adaptive threshold)"""
        if not self._noise_calibrated:
            return False
        threshold = max(self._MIN_THRESHOLD, self._noise_floor * self._SPEECH_MULTIPLIER)
        return rms > threshold

    async def listen_for_wake_word(self) -> WakeWordResult:
        """
        Block until wake word detected.

        Returns WakeWordResult with the detected phrase and any trailing text
        (e.g. "hey jarvis show me issues" -> phrase="hey jarvis", trailing="show me issues").
        """
        self._should_stop = False
        # Keep calibration across calls -- only recalibrate on first invocation
        if not self._noise_calibrated:
            self._noise_samples = []

        while not self._should_stop:
            start_time = time.time()

            # Open mic stream
            self.audio_input.start_stream()

            try:
                # --- Phase 0: Calibrate noise floor (first ~1s) ---
                if not self._noise_calibrated:
                    if self.debug:
                        print("[Wake] Calibrating noise floor...")
                    for _ in range(self._CALIBRATION_FRAMES):
                        chunk = await self.audio_input.read_chunk_async()
                        rms = self._compute_rms(chunk)
                        self._noise_samples.append(rms)

                    self._noise_floor = float(np.median(self._noise_samples))
                    self._noise_calibrated = True
                    threshold = max(self._MIN_THRESHOLD,
                                    self._noise_floor * self._SPEECH_MULTIPLIER)
                    if self.debug:
                        print("[Wake] Noise floor: %.1f, threshold: %.1f" % (
                            self._noise_floor, threshold))

                # --- Phase 1: Wait for speech onset (energy spike) ---
                audio_frames = []
                speech_detected = False

                while not self._should_stop:
                    chunk = await self.audio_input.read_chunk_async()
                    rms = self._compute_rms(chunk)

                    if self._is_speech_energy(rms):
                        # Energy spike! Start buffering
                        speech_detected = True
                        audio_frames.append(chunk)
                        if self.debug:
                            print("[Wake] Speech onset: energy=%.1f" % rms)
                        break

                    # Slowly update noise floor during silence
                    self._noise_floor = (self._noise_floor * 0.95) + (rms * 0.05)

                if not speech_detected or self._should_stop:
                    continue

                # --- Phase 2: Buffer audio until speech ends or burst timeout ---
                buffer_start = time.time()
                silence_chunks = 0
                SILENCE_CHUNKS_TO_STOP = 6  # ~384ms of silence = speech ended (was 12)

                while (time.time() - buffer_start) < self.burst_duration:
                    if self._should_stop:
                        break

                    chunk = await self.audio_input.read_chunk_async()
                    audio_frames.append(chunk)
                    rms = self._compute_rms(chunk)

                    if self._is_speech_energy(rms):
                        silence_chunks = 0
                    else:
                        silence_chunks += 1
                        if silence_chunks >= SILENCE_CHUNKS_TO_STOP:
                            if self.debug:
                                print("[Wake] Speech ended after %.1fs" % (
                                    time.time() - buffer_start))
                            break

                if self._should_stop:
                    continue

                # --- Phase 3: Transcribe via Deepgram REST ---
                audio_bytes = b''.join(audio_frames)

                # Skip if too short (< 0.2s = probably just a bump)
                min_bytes = int(0.2 * 16000 * 2)  # 0.2s at 16kHz, 16-bit
                if len(audio_bytes) < min_bytes:
                    if self.debug:
                        print("[Wake] Too short (%.0fms), skipping" % (
                            len(audio_bytes) / (16000 * 2) * 1000))
                    continue

                if self.debug:
                    duration_ms = len(audio_bytes) / (16000 * 2) * 1000
                    print("[Wake] Transcribing %.0fms of audio..." % duration_ms)

                transcript = await self.stt.transcribe_audio(audio_bytes)

                if not transcript or not transcript.strip():
                    if self.debug:
                        print("[Wake] Empty transcript, continuing")
                    continue

                if self.debug:
                    print("[Wake] Heard: '%s'" % transcript)

                # --- Phase 4: Check for wake phrase ---
                phrase, trailing = self._extract_wake_phrase(transcript)

                if phrase:
                    detection_ms = (time.time() - start_time) * 1000
                    if self.debug:
                        print("[Wake] MATCHED '%s' (%.0fms)" % (phrase, detection_ms))
                    return WakeWordResult(
                        phrase=phrase,
                        trailing_text=trailing,
                        confidence=0.9,
                        detection_time_ms=detection_ms,
                    )

                if self.debug:
                    print("[Wake] No wake phrase in '%s', continuing" % transcript)

            finally:
                self.audio_input.stop_stream()

        # Should never reach here unless stopped
        return WakeWordResult(
            phrase="",
            trailing_text="",
            confidence=0.0,
            detection_time_ms=0.0,
        )

    def _extract_wake_phrase(self, text: str) -> tuple:
        """
        Check if text starts with a wake phrase.

        Returns (phrase, remaining_text) or (None, "") if no wake phrase found.
        """
        lower = text.lower().strip()

        # Strip leading punctuation/whitespace that Deepgram sometimes adds
        lower = lower.lstrip('.,!? ')

        # Remove internal punctuation for matching (Deepgram adds commas/periods:
        # "Hey, Jarvis." -> "hey jarvis" for comparison)
        normalized = re.sub(r'[.,!?;:\'"]+', '', lower).strip()
        normalized = re.sub(r'\s+', ' ', normalized)

        # Sort by length descending so "hey jarvis" matches before "jarvis"
        sorted_phrases = sorted(self.wake_phrases, key=len, reverse=True)

        for phrase in sorted_phrases:
            if normalized.startswith(phrase):
                # Extract remaining text after the wake phrase from original
                # Find where the phrase content ends in the original lower text
                remaining = normalized[len(phrase):].strip()
                return phrase, remaining

        return None, ""

    def stop(self):
        """Stop listening for wake word"""
        self._should_stop = True
