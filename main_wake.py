"""main_wake.py - Phase 3 LUCIA: Hands-free voice assistant with continuous wake word detection
and push-to-talk manual interrupt.

Usage:
    python main_wake.py
"""

import collections
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
import wakeword
from main import run_turn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class PlaybackInterrupted(Exception):
    """Raised when user presses manual push-to-talk key to interrupt turn."""
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


def check_ollama_server() -> None:
    """Check if the local Ollama service is reachable, warning if offline."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434", timeout=1.0) as resp:
            if resp.status != 200:
                print("Warning: Ollama server returned non-200 status code.")
    except Exception:
        print("Warning: Ollama server is not running on http://localhost:11434.")
        print("Please start Ollama in a separate terminal ('ollama serve') for responses.\n")


def main():
    print("=" * 60)
    print("LUCIA Voice Assistant Online (Hands-Free Wake Word Mode)")
    print(f"LLM: Ollama ({config.MODEL})")
    print(f"STT: faster-whisper ({config.WHISPER_MODEL_SIZE})")
    print(f"TTS: Piper ({config.PIPER_VOICE_MODEL})")
    print(f"Wake Word: \"{config.WAKE_WORD_MODEL.replace('_', ' ').title()}\" (Threshold: {config.WAKE_WORD_THRESHOLD:.2f})")
    print(f"Manual Interrupt / Push-to-Talk: [{config.PUSH_TO_TALK_KEY.upper()}] (Press anytime to speak/interrupt)")
    print("Press Ctrl+C to exit.")
    print("=" * 60 + "\n")

    check_audio_devices()
    check_ollama_server()

    # Pre-load models at startup
    print("Initializing models...")
    detector = wakeword.get_wakeword_detector()
    stt.get_whisper_model()
    tts.get_piper_voice()
    speaker = tts.SentenceSpeaker()

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    # Audio queue & streaming synchronization
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    is_playback_active = threading.Event()

    # Frame size for openWakeWord: 1280 samples = 80ms at 16kHz
    chunk_samples = wakeword.WAKE_FRAME_SIZE

    def audio_input_callback(indata, frames, time_info, status):
        """Callback invoked by sounddevice for incoming microphone frames."""
        if not is_playback_active.is_set():
            audio_queue.put(indata.copy().flatten())

    try:
        mic_stream = sd.InputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="float32",
            blocksize=chunk_samples,
            callback=audio_input_callback,
        )
        mic_stream.start()
    except Exception as e:
        print(f"\nFatal Error: Failed to open microphone stream ({e}).")
        print("Please verify that a microphone is plugged in and enabled in Windows sound settings.")
        return

    # Pre-roll ring buffer: retains the last WAKE_PRE_ROLL_SECONDS of audio
    pre_roll_chunks = max(1, int(config.WAKE_PRE_ROLL_SECONDS / (chunk_samples / config.AUDIO_SAMPLE_RATE)))
    pre_roll_buffer = collections.deque(maxlen=pre_roll_chunks)

    # State machine tracking for push-to-talk manual interrupts
    STATE_IDLE = "idle"
    STATE_RECORDING = "recording"
    STATE_PROCESSING = "processing"
    STATE_PLAYBACK = "playback"

    current_state = STATE_IDLE
    state_lock = threading.Lock()

    interrupt_event = threading.Event()
    manual_trigger_event = threading.Event()

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
            with state_lock:
                if current_state in (STATE_PLAYBACK, STATE_PROCESSING):
                    # Interrupt ongoing speech or processing immediately
                    speaker.interrupt()
                    interrupt_event.set()
                elif current_state == STATE_RECORDING:
                    # Ignore extra key press while already recording (safe and idempotent)
                    pass
                elif current_state == STATE_IDLE:
                    # Manual trigger without needing the wake word
                    manual_trigger_event.set()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    pending_record = False
    is_manual_turn = False

    print(f"Listening for wake word \"{config.WAKE_WORD_MODEL.replace('_', ' ').title()}\" or press [{config.PUSH_TO_TALK_KEY.upper()}]...\n")

    try:
        while True:
            # 1. State: IDLE / WAITING (Unless arriving from an in-turn interrupt)
            if not pending_record:
                with state_lock:
                    current_state = STATE_IDLE

                if manual_trigger_event.is_set():
                    manual_trigger_event.clear()
                    is_manual_turn = True
                    print(f"\n[Manual trigger: {config.PUSH_TO_TALK_KEY.upper()}] [Listening...]", flush=True)
                else:
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # Maintain pre-roll buffer during idle listening
                    pre_roll_buffer.append(chunk)

                    # Evaluate chunk against openWakeWord model
                    is_detected, score = detector.detect_chunk(chunk)
                    if not is_detected:
                        continue

                    is_manual_turn = False
                    print(f"\n[Wake word detected! Score: {score:.2f}] [Listening...]", flush=True)
            else:
                # Arrived directly from an interrupt during playback or processing
                pending_record = False
                is_manual_turn = True

            # 2. State: RECORDING USER SPEECH (with silence-based end-of-speech detection)
            with state_lock:
                current_state = STATE_RECORDING
            interrupt_event.clear()

            recorded_chunks = list(pre_roll_buffer)  # Seed recording with pre-roll audio
            has_started_speaking = False
            speech_start_deadline = time.time() + config.WAKE_SPEECH_START_TIMEOUT
            consecutive_silence_time = 0.0
            recording_start_time = time.time()
            chunk_duration = chunk_samples / config.AUDIO_SAMPLE_RATE

            speech_timed_out = False

            while True:
                try:
                    speech_chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                recorded_chunks.append(speech_chunk)
                chunk_rms = float(np.sqrt(np.mean(speech_chunk ** 2))) if speech_chunk.size > 0 else 0.0

                # Detect active speech vs silence
                if chunk_rms >= config.WAKE_SILENCE_RMS_THRESHOLD:
                    if not has_started_speaking:
                        has_started_speaking = True
                    consecutive_silence_time = 0.0
                else:
                    if has_started_speaking:
                        consecutive_silence_time += chunk_duration

                now = time.time()

                # Case A: User never began speaking within WAKE_SPEECH_START_TIMEOUT
                if not has_started_speaking and now > speech_start_deadline:
                    print("  [No speech detected: timeout]\n")
                    speech_timed_out = True
                    break

                # Case B: User spoke, and has now paused for WAKE_END_OF_SPEECH_PAUSE
                if has_started_speaking and consecutive_silence_time >= config.WAKE_END_OF_SPEECH_PAUSE:
                    break

                # Case C: Safety max recording limit reached
                if (now - recording_start_time) >= config.WAKE_MAX_RECORDING_SECONDS:
                    break

            if speech_timed_out or not recorded_chunks:
                detector.reset()
                pre_roll_buffer.clear()
                continue

            # 3. State: PROCESSING & TRANSCRIPTION
            with state_lock:
                current_state = STATE_PROCESSING

            if interrupt_event.is_set():
                interrupt_event.clear()
                print("\n[Interrupted by key press] [Listening...]", flush=True)
                pending_record = True
                detector.reset()
                pre_roll_buffer.clear()
                continue

            audio_data = np.concatenate(recorded_chunks)
            duration = len(audio_data) / config.AUDIO_SAMPLE_RATE
            rms = float(np.sqrt(np.mean(audio_data ** 2))) if audio_data.size > 0 else 0.0

            # Amplitude filter check
            silence_thresh = config.WAKE_SILENCE_RMS_THRESHOLD
            if rms < silence_thresh:
                print(f"  [Amplitude filter] No speech detected: audio below threshold (RMS {rms:.5f} < {silence_thresh})\n")
                detector.reset()
                pre_roll_buffer.clear()
                continue

            print(f"[Processing speech... {duration:.1f}s, transcribing...]", flush=True)
            raw_transcription = stt.transcribe_raw(audio_data)

            if interrupt_event.is_set():
                interrupt_event.clear()
                print("\n[Interrupted by key press] [Listening...]", flush=True)
                pending_record = True
                detector.reset()
                pre_roll_buffer.clear()
                continue

            if not raw_transcription:
                print("  [Amplitude filter] No speech detected: audio below threshold or empty\n")
                detector.reset()
                pre_roll_buffer.clear()
                continue

            # Hallucination filter check
            if stt.is_hallucination(raw_transcription):
                print(f"  [Hallucination filter] Filtered Whisper hallucination '{raw_transcription}' as no speech\n")
                detector.reset()
                pre_roll_buffer.clear()
                continue

            # Valid speech recognized
            transcription = raw_transcription
            print(f"You: {transcription}")

            snapshot = list(messages)
            messages.append({"role": "user", "content": transcription})

            first_token = [True]

            def handle_token(token: str) -> None:
                # If interrupted by user during LLM streaming, raise to abort immediately
                # This causes the Ollama generator context manager to exit, closing the HTTP stream
                if interrupt_event.is_set():
                    raise PlaybackInterrupted("Interrupted by manual key press")
                if first_token[0]:
                    print("LUCIA: ", end="", flush=True)
                    first_token[0] = False
                print(token, end="", flush=True)
                speaker.feed_token(token)

            # 4. State: PLAYBACK (Wake detection explicitly gated off)
            with state_lock:
                current_state = STATE_PLAYBACK
            is_playback_active.set()

            interrupted_turn = False
            try:
                run_turn(messages, on_token=handle_token)
                speaker.flush()
                speaker.wait_done()
                print("\n")
            except PlaybackInterrupted:
                interrupted_turn = True
                speaker.interrupt()
                messages[:] = snapshot
                print("\n[Interrupted by key press] [Listening...]\n", flush=True)
            except KeyboardInterrupt:
                speaker.interrupt()
                messages[:] = snapshot
                print("\n\n[Interrupted turn]\n")
            except Exception as e:
                speaker.interrupt()
                messages[:] = snapshot
                err_str = str(e)
                if "10061" in err_str or "ConnectError" in type(e).__name__:
                    print("\n[Error] Cannot connect to Ollama. Is the server running? Start it with 'ollama serve'.\n")
                else:
                    print(f"\n[Error during turn: {e}]\n")

            is_playback_active.clear()

            # If user interrupted during playback, jump directly to recording user's new speech
            if interrupted_turn:
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break
                pre_roll_buffer.clear()
                detector.reset()
                pending_record = True
                continue

            # 5. State: ACOUSTIC COOLDOWN (Skipped if turn was triggered manually)
            if not is_manual_turn:
                time.sleep(config.WAKE_COOLDOWN_SECONDS)

            # Purge any queued audio frames captured during playback / cooldown
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break

            pre_roll_buffer.clear()
            detector.reset()

            print(f"Listening for wake word \"{config.WAKE_WORD_MODEL.replace('_', ' ').title()}\" or press [{config.PUSH_TO_TALK_KEY.upper()}]...\n")

    except KeyboardInterrupt:
        print("\n\nShutting down LUCIA Wake Word Assistant. Goodbye.")
    finally:
        is_playback_active.set()
        speaker.interrupt()
        mic_stream.stop()
        mic_stream.close()
        listener.stop()


if __name__ == "__main__":
    main()
