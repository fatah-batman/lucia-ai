"""tts.py - Local Text-to-Speech using Piper on CPU."""

import logging
import os
import queue
import re
import threading
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
from piper import PiperVoice

import config

logger = logging.getLogger(__name__)

_voice_instance: Optional[PiperVoice] = None
_voice_lock = threading.Lock()

# Decimal-safe sentence splitter: matches sentence-ending punctuation (.!?)
# only when followed by whitespace and a capital letter, or followed by newline.
# This prevents splitting numbers like "23.5°C" or "0.1 mm".
SENTENCE_BOUNDARY_REGEX = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+')


def ensure_voice_model_downloaded(
    model_path: str = config.PIPER_VOICE_MODEL,
    config_path: str = config.PIPER_VOICE_CONFIG,
    model_url: str = config.PIPER_VOICE_URL,
    config_url: str = config.PIPER_CONFIG_URL,
) -> None:
    """Download the Piper voice model and JSON config if not already present."""
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

    files_to_download = [
        (model_path, model_url, "voice ONNX model (~63 MB)"),
        (config_path, config_url, "voice config JSON (~5 KB)"),
    ]

    for path, url, desc in files_to_download:
        if not os.path.exists(path):
            print(f"Downloading Piper {desc}...")
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {path}")


def get_piper_voice() -> PiperVoice:
    """Load and return the singleton PiperVoice instance."""
    global _voice_instance
    if _voice_instance is None:
        with _voice_lock:
            if _voice_instance is None:
                ensure_voice_model_downloaded()
                if not os.path.exists(config.PIPER_VOICE_MODEL):
                    raise FileNotFoundError(
                        f"Piper model not found at {config.PIPER_VOICE_MODEL}."
                    )
                _voice_instance = PiperVoice.load(
                    config.PIPER_VOICE_MODEL,
                    config_path=config.PIPER_VOICE_CONFIG,
                )
    return _voice_instance


def synthesize_text_to_audio(text: str) -> tuple[np.ndarray, int]:
    """Synthesize text into a single float32 NumPy audio array and return (audio, sample_rate)."""
    voice = get_piper_voice()
    chunks = list(voice.synthesize(text))
    if not chunks:
        return np.array([], dtype=np.float32), 22050

    sample_rate = chunks[0].sample_rate
    audio_arrays = [chunk.audio_float_array for chunk in chunks if chunk.audio_float_array is not None]
    if not audio_arrays:
        return np.array([], dtype=np.float32), sample_rate

    combined = np.concatenate(audio_arrays)
    return combined, sample_rate


def speak(text: str) -> None:
    """Synthesize and play speech directly through the default audio output.

    Args:
        text: The text string to speak.
    """
    clean_text = text.strip()
    if not clean_text:
        return

    audio, sample_rate = synthesize_text_to_audio(clean_text)
    if len(audio) > 0:
        sd.play(audio, samplerate=sample_rate)
        sd.wait()


class SentenceSpeaker:
    """Streams incoming tokens, batches them into complete sentences, and speaks concurrently.

    Features:
      - Uses decimal-safe sentence boundaries (preserves numbers like 23.5°C).
      - Runs playback in a background worker thread so the LLM generation is never blocked.
      - Supports immediate interruption (barge-in) when the user speaks.
    """

    def __init__(self):
        self.buffer = ""
        self.queue: queue.Queue[Optional[str]] = queue.Queue()
        self.interrupt_flag = threading.Event()
        self.is_speaking_flag = threading.Event()
        self._worker_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._worker_thread.start()

    def _playback_loop(self) -> None:
        """Background worker consuming and speaking sentences from the queue."""
        while True:
            sentence = self.queue.get()
            if sentence is None:  # Shutdown signal if needed
                break

            if self.interrupt_flag.is_set():
                self.queue.task_done()
                continue

            self.is_speaking_flag.set()
            try:
                audio, sample_rate = synthesize_text_to_audio(sentence)
                if len(audio) > 0 and not self.interrupt_flag.is_set():
                    sd.play(audio, samplerate=sample_rate)
                    while sd.get_stream().active:
                        if self.interrupt_flag.is_set():
                            sd.stop()
                            break
                        sd.sleep(30)
            except Exception as e:
                logger.error(f"TTS playback error: {e}")
            finally:
                self.is_speaking_flag.clear()
                self.queue.task_done()

    def feed_token(self, token: str) -> None:
        """Receive a token from LLM stream, detect complete sentences, and enqueue them."""
        self.buffer += token
        while True:
            match = SENTENCE_BOUNDARY_REGEX.search(self.buffer)
            if not match:
                break
            # Split off the complete sentence
            sentence = self.buffer[: match.start() + 1].strip()
            self.buffer = self.buffer[match.end() :]
            if sentence and not self.interrupt_flag.is_set():
                self.queue.put(sentence)

    def flush(self) -> None:
        """Flush any remaining text in the buffer to the playback queue."""
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining and not self.interrupt_flag.is_set():
            self.queue.put(remaining)

    def interrupt(self) -> None:
        """Immediately stop ongoing playback and clear queued sentences (barge-in)."""
        self.interrupt_flag.set()
        # Empty the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
        sd.stop()
        self.buffer = ""
        self.is_speaking_flag.clear()
        # Reset interrupt flag for future turns
        self.interrupt_flag.clear()

    def wait_done(self) -> None:
        """Wait until all queued sentences have finished playing."""
        self.queue.join()

    def is_playing(self) -> bool:
        """Return True if audio is actively playing or queued."""
        return self.is_speaking_flag.is_set() or not self.queue.empty()
