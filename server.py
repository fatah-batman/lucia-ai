"""server.py - FastAPI + WebSocket server for the LUCIA Web Dashboard & 3D Reactive Orb.

Usage:
    python server.py
"""

import asyncio
import collections
import contextlib
import ctypes
import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lucia-server")


class PlaybackInterrupted(Exception):
    """Raised when user triggers manual interrupt."""
    pass


# ---------------------------------------------------------------------------
# WebSocket Connection & Event Manager
# ---------------------------------------------------------------------------
class EventManager:
    """Thread-safe WebSocket manager broadcasting structured events from voice thread to web clients."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.current_state: str = "idle"
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)
        # Send current assistant state immediately on connection
        await websocket.send_json({"type": "state", "value": self.current_state})

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    def set_state(self, state: str):
        self.current_state = state
        self.broadcast_sync({"type": "state", "value": state})

    def broadcast_sync(self, message: dict):
        """Thread-safe dispatch of JSON message to all connected WebSockets."""
        with self._lock:
            if not self.active_connections or self.loop is None or self.loop.is_closed():
                return
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def _broadcast(self, message: dict):
        with self._lock:
            targets = list(self.active_connections)
        for connection in targets:
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = EventManager()

# Global events for inter-thread synchronization
interrupt_event = threading.Event()
manual_trigger_event = threading.Event()
shutdown_event = threading.Event()


def is_terminal_focused() -> bool:
    """Return True if terminal has foreground focus (Windows only)."""
    if not config.REQUIRE_TERMINAL_FOCUS or sys.platform != "win32":
        return True
    try:
        console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if console_hwnd == 0:
            return True
        active_hwnd = ctypes.windll.user32.GetForegroundWindow()
        return active_hwnd == console_hwnd
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Background Voice Pipeline Loop (Reusing wakeword, stt, tts, run_turn)
# ---------------------------------------------------------------------------
def voice_pipeline_worker():
    """Runs the hands-free wake word listening loop and streams events to the dashboard."""
    logger.info("Initializing voice pipeline components...")

    detector = wakeword.get_wakeword_detector()
    stt.get_whisper_model()
    tts.get_piper_voice()

    def on_tts_amplitude(rms: float):
        # Scale RMS for dynamic orb ripples
        scaled = min(1.0, float(rms * 4.5))
        manager.broadcast_sync({"type": "amplitude", "value": scaled})

    speaker = tts.SentenceSpeaker(on_playback_amplitude=on_tts_amplitude)
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    is_playback_active = threading.Event()
    chunk_samples = wakeword.WAKE_FRAME_SIZE

    def audio_input_callback(indata, frames, time_info, status):
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
        logger.error(f"Failed to open microphone stream: {e}")
        return

    pre_roll_chunks = max(1, int(config.WAKE_PRE_ROLL_SECONDS / (chunk_samples / config.AUDIO_SAMPLE_RATE)))
    pre_roll_buffer = collections.deque(maxlen=pre_roll_chunks)

    # State tracking
    STATE_IDLE = "idle"
    STATE_RECORDING = "recording"
    STATE_PROCESSING = "processing"
    STATE_PLAYBACK = "speaking"

    current_state = STATE_IDLE
    state_lock = threading.Lock()

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

    def on_key_press(key):
        if not is_terminal_focused():
            return
        if is_target_key(key):
            with state_lock:
                if current_state in (STATE_PLAYBACK, STATE_PROCESSING):
                    speaker.interrupt()
                    interrupt_event.set()
                elif current_state == STATE_IDLE:
                    manual_trigger_event.set()

    key_listener = keyboard.Listener(on_press=on_key_press)
    key_listener.start()

    pending_record = False
    is_manual_turn = False

    logger.info(f"LUCIA Voice Assistant Online. Listening for \"{config.WAKE_WORD_MODEL.replace('_', ' ').title()}\" or push-to-talk...")

    try:
        while not shutdown_event.is_set():
            # 1. State: IDLE
            if not pending_record:
                with state_lock:
                    current_state = STATE_IDLE
                manager.set_state("idle")

                if manual_trigger_event.is_set():
                    manual_trigger_event.clear()
                    is_manual_turn = True
                    print(f"\n[Manual trigger: {config.PUSH_TO_TALK_KEY.upper()}] [Listening...]", flush=True)
                else:
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    pre_roll_buffer.append(chunk)
                    is_detected, score = detector.detect_chunk(chunk)
                    if not is_detected:
                        continue

                    is_manual_turn = False
                    print(f"\n[Wake word detected! Score: {score:.2f}] [Listening...]", flush=True)
            else:
                pending_record = False
                is_manual_turn = True

            # 2. State: RECORDING (Listening to user query)
            with state_lock:
                current_state = STATE_RECORDING
            manager.set_state("listening")
            interrupt_event.clear()

            recorded_chunks = list(pre_roll_buffer)
            has_started_speaking = False
            speech_start_deadline = time.time() + config.WAKE_SPEECH_START_TIMEOUT
            consecutive_silence_time = 0.0
            recording_start_time = time.time()
            chunk_duration = chunk_samples / config.AUDIO_SAMPLE_RATE

            speech_timed_out = False

            while not shutdown_event.is_set():
                try:
                    speech_chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                recorded_chunks.append(speech_chunk)
                chunk_rms = float(np.sqrt(np.mean(speech_chunk ** 2))) if speech_chunk.size > 0 else 0.0

                # Stream live mic amplitude to orb for energetic ripples
                scaled_amp = min(1.0, float(chunk_rms * 6.0))
                manager.broadcast_sync({"type": "amplitude", "value": scaled_amp})

                if chunk_rms >= config.WAKE_SILENCE_RMS_THRESHOLD:
                    if not has_started_speaking:
                        has_started_speaking = True
                    consecutive_silence_time = 0.0
                else:
                    if has_started_speaking:
                        consecutive_silence_time += chunk_duration

                now = time.time()
                if not has_started_speaking and now > speech_start_deadline:
                    print("  [No speech detected: timeout]\n")
                    speech_timed_out = True
                    break

                if has_started_speaking and consecutive_silence_time >= config.WAKE_END_OF_SPEECH_PAUSE:
                    break

                if (now - recording_start_time) >= config.WAKE_MAX_RECORDING_SECONDS:
                    break

            # Reset mic amplitude back to zero when recording ends
            manager.broadcast_sync({"type": "amplitude", "value": 0.0})

            if speech_timed_out or not recorded_chunks:
                detector.reset()
                pre_roll_buffer.clear()
                continue

            # 3. State: PROCESSING & TRANSCRIPTION
            with state_lock:
                current_state = STATE_PROCESSING
            manager.set_state("processing")

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

            if stt.is_hallucination(raw_transcription):
                print(f"  [Hallucination filter] Filtered Whisper hallucination '{raw_transcription}' as no speech\n")
                detector.reset()
                pre_roll_buffer.clear()
                continue

            # Valid speech recognized: broadcast to web chat log
            transcription = raw_transcription
            print(f"You: {transcription}")
            manager.broadcast_sync({"type": "user_speech", "text": transcription})

            snapshot = list(messages)
            messages.append({"role": "user", "content": transcription})

            first_token = [True]

            def handle_token(token: str) -> None:
                if interrupt_event.is_set():
                    raise PlaybackInterrupted("Interrupted by user")
                if first_token[0]:
                    print("LUCIA: ", end="", flush=True)
                    first_token[0] = False
                print(token, end="", flush=True)
                manager.broadcast_sync({"type": "token", "token": token})
                speaker.feed_token(token)

            # 4. State: PLAYBACK (Speaking)
            with state_lock:
                current_state = STATE_PLAYBACK
            manager.set_state("speaking")
            is_playback_active.set()

            interrupted_turn = False
            full_reply_text = ""
            try:
                full_reply_text = run_turn(messages, on_token=handle_token)
                speaker.flush()
                speaker.wait_done()
                print("\n")
                manager.broadcast_sync({"type": "turn_done", "content": full_reply_text})
            except PlaybackInterrupted:
                interrupted_turn = True
                speaker.interrupt()
                messages[:] = snapshot
                print("\n[Interrupted by key press] [Listening...]\n", flush=True)
                manager.broadcast_sync({"type": "turn_interrupted"})
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
                    manager.broadcast_sync({"type": "error", "message": "Cannot connect to local Ollama server."})
                else:
                    print(f"\n[Error during turn: {e}]\n")

            is_playback_active.clear()
            manager.broadcast_sync({"type": "amplitude", "value": 0.0})

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

            # 5. State: ACOUSTIC COOLDOWN
            if not is_manual_turn:
                time.sleep(config.WAKE_COOLDOWN_SECONDS)

            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break

            pre_roll_buffer.clear()
            detector.reset()
            print(f"Listening for wake word \"{config.WAKE_WORD_MODEL.replace('_', ' ').title()}\" or press [{config.PUSH_TO_TALK_KEY.upper()}]...\n")

    except KeyboardInterrupt:
        logger.info("Voice pipeline shutting down...")
    finally:
        is_playback_active.set()
        speaker.interrupt()
        mic_stream.stop()
        mic_stream.close()
        key_listener.stop()


# ---------------------------------------------------------------------------
# FastAPI Web Application & WebSocket Routes
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Capture running asyncio loop for thread-safe WebSocket broadcasts
    loop = asyncio.get_running_loop()
    manager.set_loop(loop)

    # Launch background voice pipeline thread
    voice_thread = threading.Thread(target=voice_pipeline_worker, daemon=True)
    voice_thread.start()
    yield
    shutdown_event.set()


app = FastAPI(title="LUCIA Web Dashboard", lifespan=lifespan)

# Mount static folder
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "LUCIA Dashboard initialized. Place index.html in static/"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                # Allow triggering interrupt / manual speech from web UI button or browser hotkey
                if msg.get("action") == "interrupt":
                    speaker_instance = tts.SentenceSpeaker()
                    speaker_instance.interrupt()
                    interrupt_event.set()
                    manual_trigger_event.set()
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


def main():
    print("=" * 60)
    print("Starting LUCIA Web Dashboard & Voice Assistant")
    print("Open http://127.0.0.1:8000 in your browser to view the orb")
    print("=" * 60)
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False, log_level="warning")


if __name__ == "__main__":
    main()
