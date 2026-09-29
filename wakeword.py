"""wakeword.py - Continuous wake word detection using openWakeWord on CPU."""

import logging
import os
import sys
from typing import Optional, Tuple, Union

import numpy as np

import config

logger = logging.getLogger(__name__)

# Standard openWakeWord input frame size (80ms at 16kHz = 1280 samples)
WAKE_FRAME_SIZE = 1280


class WakeWordDetector:
    """Detects spoken wake words from audio frames using openWakeWord (CPU/ONNX)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.model_name = model_name or config.WAKE_WORD_MODEL
        self.threshold = threshold if threshold is not None else config.WAKE_WORD_THRESHOLD
        self.model = self._load_model()

    def _load_model(self):
        """Load the openWakeWord ONNX model, downloading weights if necessary."""
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as e:
            msg = (
                "\n[Error] 'openwakeword' package is not installed.\n"
                "Please run: pip install openwakeword\n"
            )
            print(msg, file=sys.stderr)
            raise RuntimeError(msg) from e

        # Check if model exists or needs download
        try:
            model = Model(
                wakeword_models=[self.model_name],
                inference_framework="onnx",
            )
            return model
        except Exception as load_err:
            logger.info(f"Model '{self.model_name}' not loaded immediately ({load_err}). Attempting download...")
            try:
                print(f"Downloading openWakeWord model '{self.model_name}'...")
                openwakeword.utils.download_models([self.model_name])
                model = Model(
                    wakeword_models=[self.model_name],
                    inference_framework="onnx",
                )
                print(f"openWakeWord model '{self.model_name}' downloaded and loaded successfully.")
                return model
            except Exception as dl_err:
                error_msg = (
                    f"\n[Error] Failed to load or download openWakeWord model '{self.model_name}'.\n"
                    "Please check your internet connection for the initial model download.\n"
                    f"Error details: {dl_err}\n"
                )
                print(error_msg, file=sys.stderr)
                raise RuntimeError(error_msg) from dl_err

    def reset(self) -> None:
        """Reset the internal sliding window feature buffer of the model."""
        if self.model is not None:
            self.model.reset()

    def detect_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """Process an audio chunk and determine if the wake word was detected.

        Args:
            audio_chunk: 1D NumPy array of audio samples (16 kHz).
                         Can be float32 (-1.0 to 1.0) or int16 (-32768 to 32767).
                         Ideally WAKE_FRAME_SIZE (1280 samples / 80ms).

        Returns:
            Tuple of (is_detected: bool, score: float).
        """
        if self.model is None or audio_chunk is None or audio_chunk.size == 0:
            return False, 0.0

        # Convert float32 to int16 if needed
        if np.issubdtype(audio_chunk.dtype, np.floating):
            pcm16 = np.clip(audio_chunk * 32767.0, -32768.0, 32767.0).astype(np.int16)
        elif audio_chunk.dtype == np.int16:
            pcm16 = audio_chunk
        else:
            pcm16 = audio_chunk.astype(np.int16)

        if pcm16.ndim > 1:
            pcm16 = pcm16.flatten()

        try:
            # Predict returns a dict, e.g. {"hey_jarvis": 0.0} or {"hey_jarvis_v0.1": 0.85}
            prediction = self.model.predict(pcm16)

            # Match model key
            score = 0.0
            for key, val in prediction.items():
                if self.model_name in key or key in self.model_name:
                    score = float(val)
                    break
            else:
                # If key not explicitly matched, take the highest score from the prediction dict
                if prediction:
                    score = float(max(prediction.values()))

            is_detected = score >= self.threshold
            return is_detected, score
        except Exception as e:
            logger.error(f"Error during wake word inference: {e}")
            return False, 0.0


_detector_instance: Optional[WakeWordDetector] = None


def get_wakeword_detector() -> WakeWordDetector:
    """Return the singleton WakeWordDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = WakeWordDetector()
    return _detector_instance
