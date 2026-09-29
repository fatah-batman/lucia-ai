"""main_voice.py - Phase 2 LUCIA: Push-to-talk voice assistant with local STT and streaming TTS.

Usage:
    python main_voice.py
"""

import ctypes
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
from pynput import keyboard

import config
import stt
import tts
from main import run_turn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def is_terminal_focused() -> bool:
    """Return True if the console/terminal window currently has foreground focus (Windows only)."""
    if not config.REQUIRE_TERMINAL_FOCUS or sys.platform != "win32":
        return True

    try:
        console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if console_hwnd == 0:
            return True  # Running under an IDE terminal emulator, allow hotkey
        active_hwnd = ctypes.windll.user32.GetForegroundWindow()
        return active_hwnd == console_hwnd
    except Exception:
        return True


def check_audio_devices() -> None:
    """Check that an input microphone and output playback device exist."""
    try:
        devices = sd.query_devices()
        input_dev = sd.default.device[0]
        output_dev = sd.default.device[1]
        if input_dev is None or input_dev < 0:
            print("Warning: No default microphone device detected by sounddevice.")
        if output_dev is None or output_dev < 0:
            print("Warning: No default speaker device detected by sounddevice.")
    except Exception as e:
        print(f"Warning: Audio device check failed ({e}). Check your audio hardware.")


def main():
    print("=" * 60)
    print(f"LUCIA Voice Assistant Online (Model: {config.MODEL})")
    print(f"STT: faster-whisper ({config.WHISPER_MODEL_SIZE})")
    print(f"TTS: Piper ({config.PIPER_VOICE_MODEL})")
    print(f"Push-to-talk key: [{config.PUSH_TO_TALK_KEY.upper()}] (Hold to speak, release to send)")
    print("Press Ctrl+C to exit.")
    print("=" * 60 + "\n")

    check_audio_devices()

    # Pre-load STT and TTS models once at startup
    stt.get_whisper_model()
    tts.get_piper_voice()
    speaker = tts.SentenceSpeaker()

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    # Push-to-talk audio recording state
    is_recording = threading.Event()
    recorded_chunks: list[np.ndarray] = []
    audio_lock = threading.Lock()
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def audio_input_callback(indata, frames, time_info, status):
        """Callback invoked by sounddevice for incoming microphone frames."""
        if is_recording.is_set():
            with audio_lock:
                recorded_chunks.append(indata.copy().flatten())

    # Open continuous microphone input stream (16 kHz, mono, float32)
    try:
        mic_stream = sd.InputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="float32",
            callback=audio_input_callback,
        )
        mic_stream.start()
    except Exception as e:
        print(f"\nFatal Error: Failed to open microphone stream ({e}).")
        print("Please verify that a microphone is plugged in and enabled in Windows sound settings.")
        return

    # Key matching helper
    def is_target_key(key) -> bool:
        target = config.PUSH_TO_TALK_KEY.lower()
        if target == "space":
            return (
                key == keyboard.Key.space
                or getattr(key, "name", None) == "space"
                or getattr(key, "char", None) == " "
            )
        try:
            return (
                getattr(key, "name", None) == target
                or getattr(key, "char", "").lower() == target
            )
        except Exception:
            return False

    def on_press(key):
        if not is_terminal_focused():
            return

        if is_target_key(key):
            # Barge-in: if LUCIA is currently speaking, halt playback immediately
            if speaker.is_playing():
                speaker.interrupt()

            if not is_recording.is_set():
                with audio_lock:
                    recorded_chunks.clear()
                is_recording.set()
                print(f"\n[● Recording... speak now, release {config.PUSH_TO_TALK_KEY.upper()} to send] ", end="", flush=True)

    def on_release(key):
        if is_target_key(key) and is_recording.is_set():
            is_recording.clear()
            with audio_lock:
                if recorded_chunks:
                    audio_data = np.concatenate(recorded_chunks)
                    audio_queue.put(audio_data)
                else:
                    audio_queue.put(np.array([], dtype=np.float32))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print(f"Ready! Hold [{config.PUSH_TO_TALK_KEY.upper()}] to speak, release to send.\n")

    try:
        while True:
            try:
                audio_data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            duration = len(audio_data) / config.AUDIO_SAMPLE_RATE

            # Ignore brief taps
            if duration < config.AUDIO_MIN_DURATION_SECONDS:
                print(f"\n[Tap ignored: held for {duration:.2f}s < minimum {config.AUDIO_MIN_DURATION_SECONDS}s]\n")
                continue

            # Check RMS energy threshold
            rms = float(np.sqrt(np.mean(audio_data ** 2)))
            if rms < config.AUDIO_RMS_THRESHOLD:
                print(f"\n[No speech detected: audio too quiet (RMS: {rms:.4f} < {config.AUDIO_RMS_THRESHOLD})]\n")
                continue

            print(f"\n[Processing speech... {duration:.1f}s, transcribing...]", flush=True)
            transcription = stt.transcribe(audio_data)

            if not transcription:
                print("[No recognizable speech detected]\n")
                continue

            print(f"You: {transcription}")

            snapshot = list(messages)
            messages.append({"role": "user", "content": transcription})

            first_token = [True]

            def handle_token(token: str) -> None:
                if first_token[0]:
                    print("LUCIA: ", end="", flush=True)
                    first_token[0] = False
                print(token, end="", flush=True)
                speaker.feed_token(token)

            try:
                run_turn(messages, on_token=handle_token)
                speaker.flush()
                speaker.wait_done()
                print("\n")
            except KeyboardInterrupt:
                speaker.interrupt()
                messages[:] = snapshot
                print("\n\n[Interrupted turn]\n")
                continue

    except KeyboardInterrupt:
        print("\n\nShutting down LUCIA Voice. Goodbye.")
    finally:
        speaker.interrupt()
        mic_stream.stop()
        mic_stream.close()
        listener.stop()


if __name__ == "__main__":
    main()
