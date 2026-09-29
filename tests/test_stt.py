"""tests/test_stt.py - Unit tests for Speech-to-Text module."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import stt


def test_transcribe_empty_audio_returns_empty_string():
    """Empty numpy array should immediately return empty string without invoking model."""
    empty_audio = np.array([], dtype=np.float32)
    result = stt.transcribe(empty_audio)
    assert result == ""


def test_transcribe_silence_below_rms_returns_empty_string():
    """Audio below the RMS energy threshold should be rejected as silence/noise."""
    # Create very low amplitude noise (RMS well below 0.015)
    silent_audio = np.zeros(16000, dtype=np.float32)
    result = stt.transcribe(silent_audio)
    assert result == ""


@patch("stt.get_whisper_model")
def test_transcribe_valid_audio_returns_text(mock_get_model):
    """Verify transcribe processes non-silent audio and formats segments into text."""
    mock_segment = MagicMock()
    mock_segment.text = "what time is it"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)
    mock_get_model.return_value = mock_model

    # Simulated audio buffer with sufficient amplitude
    audio_data = np.full(16000, 0.1, dtype=np.float32)
    result = stt.transcribe(audio_data)

    assert result == "what time is it"
    assert mock_model.transcribe.called


@patch("stt.WhisperModel")
def test_whisper_automatic_gpu_to_cpu_fallback(mock_whisper_class):
    """Verify that if CUDA GPU initialization fails, it automatically falls back to CPU."""
    # First call (CUDA) raises exception, second call (CPU) succeeds
    mock_whisper_class.side_effect = [
        Exception("CUDA driver library cublas64_12.dll not found"),
        MagicMock(),
    ]

    with patch("stt._whisper_instance", None):
        with patch("config.WHISPER_DEVICE", "cuda"):
            model = stt.get_whisper_model()
            assert stt.get_active_device() == "cpu"
            assert mock_whisper_class.call_count == 2
            # Second call should specify device="cpu"
            call_kwargs = mock_whisper_class.call_args_list[1][1]
            assert call_kwargs["device"] == "cpu"
            assert call_kwargs["compute_type"] == "int8"
