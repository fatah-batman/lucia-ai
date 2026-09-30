"""tests/test_wakeword.py - Unit tests for wake word detection module."""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import config
import wakeword


def test_wakeword_detector_initialization():
    """Verify WakeWordDetector initializes with configured model and threshold."""
    detector = wakeword.WakeWordDetector(
        model_name=config.WAKE_WORD_MODEL,
        threshold=config.WAKE_WORD_THRESHOLD,
    )
    assert detector.model_name == config.WAKE_WORD_MODEL
    assert detector.threshold == config.WAKE_WORD_THRESHOLD
    assert detector.model is not None


def test_wakeword_detector_detects_silence_as_negative():
    """Feeding silence (zeros) must not trigger wake word detection."""
    detector = wakeword.get_wakeword_detector()
    detector.reset()

    silent_chunk = np.zeros(wakeword.WAKE_FRAME_SIZE, dtype=np.int16)
    is_detected, score = detector.detect_chunk(silent_chunk)

    assert is_detected is False
    assert score < detector.threshold


def test_wakeword_detector_detects_low_ambient_noise_as_negative():
    """Feeding low-amplitude random noise must not trigger wake word detection."""
    detector = wakeword.get_wakeword_detector()
    detector.reset()

    # Low amplitude noise in int16 range [-50, 50]
    rng = np.random.default_rng(42)
    noise_chunk = rng.integers(-50, 50, size=wakeword.WAKE_FRAME_SIZE, dtype=np.int16)

    is_detected, score = detector.detect_chunk(noise_chunk)
    assert is_detected is False
    assert score < detector.threshold


def test_wakeword_detector_empty_or_none_chunk():
    """Empty or None audio chunks should return (False, 0.0) without crashing."""
    detector = wakeword.get_wakeword_detector()
    assert detector.detect_chunk(None) == (False, 0.0)
    assert detector.detect_chunk(np.array([], dtype=np.int16)) == (False, 0.0)


def test_wakeword_detector_converts_float32_safely():
    """Float32 audio arrays (-1.0 to 1.0) must be automatically converted to int16."""
    detector = wakeword.get_wakeword_detector()
    detector.reset()

    float_silence = np.zeros(wakeword.WAKE_FRAME_SIZE, dtype=np.float32)
    is_detected, score = detector.detect_chunk(float_silence)

    assert is_detected is False
    assert score < detector.threshold


def test_wakeword_detector_reset():
    """Verify reset() executes cleanly without exception."""
    detector = wakeword.get_wakeword_detector()
    detector.reset()


def test_wakeword_detector_threshold_logic():
    """Verify threshold boundary logic using controlled prediction outputs."""
    detector = wakeword.WakeWordDetector(threshold=0.5)

    dummy_chunk = np.zeros(wakeword.WAKE_FRAME_SIZE, dtype=np.int16)

    # Mock prediction score below threshold (0.49)
    with patch.object(detector.model, "predict", return_value={"hey_jarvis": 0.49}):
        detected, score = detector.detect_chunk(dummy_chunk)
        assert detected is False
        assert score == 0.49

    # Mock prediction score at threshold (0.50)
    with patch.object(detector.model, "predict", return_value={"hey_jarvis": 0.50}):
        detected, score = detector.detect_chunk(dummy_chunk)
        assert detected is True
        assert score == 0.50

    # Mock prediction score above threshold (0.85)
    with patch.object(detector.model, "predict", return_value={"hey_jarvis": 0.85}):
        detected, score = detector.detect_chunk(dummy_chunk)
        assert detected is True
        assert score == 0.85


def test_main_wake_interrupt_exception():
    """Verify PlaybackInterrupted is properly defined and catchable as an Exception."""
    import main_wake
    assert issubclass(main_wake.PlaybackInterrupted, Exception)



def test_wakeword_acoustic_sample_real_utterance():
    """Tier 2: Real acoustic validation test.

    If a real recorded WAV file of 'Hey Jarvis' is provided at tests/audio/hey_jarvis_sample.wav,
    this test will run and verify that the acoustic model detects it.
    If no sample file is present, it is cleanly skipped with an informative notice.
    """
    sample_path = os.path.join(os.path.dirname(__file__), "audio", "hey_jarvis_sample.wav")
    if not os.path.exists(sample_path):
        pytest.skip(
            "No real acoustic sample provided at tests/audio/hey_jarvis_sample.wav. "
            "Skipping genuine acoustic recognition test."
        )

    import wave
    with wave.open(sample_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    assert framerate == 16000, f"Acoustic sample must be 16kHz, found {framerate}Hz"
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

    detector = wakeword.WakeWordDetector(threshold=0.5)
    detector.reset()

    detected_any = False
    for i in range(0, len(audio_int16) - wakeword.WAKE_FRAME_SIZE + 1, wakeword.WAKE_FRAME_SIZE):
        chunk = audio_int16[i : i + wakeword.WAKE_FRAME_SIZE]
        is_detected, _ = detector.detect_chunk(chunk)
        if is_detected:
            detected_any = True
            break

    assert detected_any is True, "Model failed to detect 'Hey Jarvis' in the real acoustic test sample"
