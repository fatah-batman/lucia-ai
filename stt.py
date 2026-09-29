"""stt.py - Local Speech-to-Text using faster-whisper on GPU with automatic CPU fallback."""

import logging
import re
import threading
from typing import Optional, Union

import numpy as np
from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)

_whisper_instance: Optional[WhisperModel] = None
_whisper_lock = threading.Lock()
_device_used: str = "unknown"


def is_hallucination(text: str) -> bool:
    """Check if transcribed text is empty or matches known Whisper silence hallucinations.

    Strips punctuation and whitespace, lowercases the text, and checks against
    config.WHISPER_HALLUCINATION_PHRASES.
    """
    if not text:
        return True

    # Strip punctuation and whitespace, lowercase
    cleaned = re.sub(r"[^\w\s]", "", text).strip().lower()
    if not cleaned:
        return True

    normalized_hallucinations = {
        re.sub(r"[^\w\s]", "", phrase).strip().lower()
        for phrase in getattr(config, "WHISPER_HALLUCINATION_PHRASES", [])
    }

    return cleaned in normalized_hallucinations


def get_whisper_model() -> WhisperModel:
    """Load and return the singleton WhisperModel instance, with automatic GPU-to-CPU fallback."""
    global _whisper_instance, _device_used
    if _whisper_instance is None:
        with _whisper_lock:
            if _whisper_instance is None:
                preferred_device = config.WHISPER_DEVICE
                compute_type = config.WHISPER_COMPUTE_TYPE
                model_size = config.WHISPER_MODEL_SIZE

                if preferred_device == "cuda":
                    try:
                        print(f"Loading Whisper model '{model_size}' on CUDA GPU ({compute_type})...")
                        candidate = WhisperModel(
                            model_size,
                            device="cuda",
                            compute_type=compute_type,
                        )
                        # Verify CUDA runtime / cuBLAS DLLs actually work on this machine
                        probe_audio = np.zeros(16000, dtype=np.float32)
                        list(candidate.transcribe(probe_audio, beam_size=1)[0])
                        _whisper_instance = candidate
                        _device_used = "cuda"
                        print("Whisper verified and running on CUDA GPU.")
                    except Exception as e:
                        print(
                            f"\nNotice: CUDA GPU execution unavailable ({e}).\n"
                            "Automatically falling back to CPU (compute_type='int8')...\n"
                        )
                        _whisper_instance = WhisperModel(
                            model_size,
                            device="cpu",
                            compute_type="int8",
                        )
                        _device_used = "cpu"
                        print("Whisper loaded and running on CPU.")
                else:
                    print(f"Loading Whisper model '{model_size}' on CPU (int8)...")
                    _whisper_instance = WhisperModel(
                        model_size,
                        device="cpu",
                        compute_type="int8",
                    )
                    _device_used = "cpu"
                    print("Whisper loaded and running on CPU.")

    return _whisper_instance


def get_active_device() -> str:
    """Return the device ('cuda' or 'cpu') that the Whisper model is running on."""
    return _device_used


def transcribe_raw(audio: Union[np.ndarray, str]) -> str:
    """Transcribe audio into text without filtering hallucinations.

    Args:
        audio: 1D float32 NumPy array (16 kHz mono) or path to an audio file.

    Returns:
        The raw transcribed text string, or empty string on error/silence.
    """
    if isinstance(audio, np.ndarray):
        if audio.size == 0:
            return ""

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.flatten()

        # Check energy / RMS threshold
        threshold = getattr(config, "SILENCE_RMS_THRESHOLD", config.AUDIO_RMS_THRESHOLD)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < threshold:
            logger.debug(f"Audio below RMS threshold ({rms:.5f} < {threshold})")
            return ""

    try:
        model = get_whisper_model()
        segments, _ = model.transcribe(
            audio,
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return ""


def transcribe(audio: Union[np.ndarray, str]) -> str:
    """Transcribe audio into text and filter out known Whisper hallucinations.

    Args:
        audio: 1D float32 NumPy array (16 kHz mono) or path to an audio file.

    Returns:
        The transcribed text string, or empty string if silence or hallucination.
    """
    raw_text = transcribe_raw(audio)
    if is_hallucination(raw_text):
        return ""
    return raw_text
