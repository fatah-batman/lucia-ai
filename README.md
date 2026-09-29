# LUCIA - Personal AI Assistant (100% Free & Local)

Inspired by JARVIS from Iron Man. Built in modular phases:
- **Phase 1**: Text assistant with Ollama tool calling, token streaming, and rolling history trimming.
- **Phase 2**: Local voice input (STT via `faster-whisper`) and output (TTS via `piper-tts`) with push-to-talk, barge-in, silence/hallucination filtering, and decimal-safe sentence-streaming speech.
- **Phase 3**: Hands-free wake word detection via `openWakeWord` ("Hey Jarvis" stand-in), silence-based end-of-speech capture, and acoustic cooldown.

---

## Setup

1. **Install Ollama**: https://ollama.com/download
2. **Pull the model** (one-time, ~4.7 GB):
   ```bash
   ollama pull qwen2.5:7b
   ```
   *(Low on RAM? Set `MODEL = "qwen2.5:3b"` or `"llama3.2:3b"` in `config.py`).*
3. **Create and activate a virtual environment** (Python 3.10+):
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run all automated unit tests**:
   ```bash
   pytest
   ```

---

## Running LUCIA

### 1. Hands-Free Wake Word Mode (Phase 3)
```bash
python main_wake.py
```
- **Hands-Free Wake Word**: Say **"Hey Jarvis"** to wake LUCIA up. No key press required.
- **End-of-Speech Detection**: Automatically stops recording when you finish speaking (after `WAKE_END_OF_SPEECH_PAUSE = 1.2` seconds of silence).
- **Pre-Roll Ring Buffer**: Captures 0.5s prior to trigger so the very first word of your command is never clipped.
- **Short Command Friendly**: Preserves quick queries (e.g. *"time?"*, *"stop"*) without hard minimum duration rejects.
- **Acoustic Cooldown**: Wake-word listening is completely gated off while LUCIA speaks, and a 1-second acoustic cooldown purges echo before listening re-arms.
- **Tuning Wake Detection**: Adjust `WAKE_WORD_THRESHOLD` (default: `0.5`) in `config.py` (increase to `0.6`–`0.65` if TV/media causes false triggers).

### 2. Push-to-Talk Voice Mode (Phase 2)
```bash
python main_voice.py
```
- **Push-to-Talk**: Hold **SPACEBAR** to speak, release to send.
- **Silence & Hallucination Filtering**: Rejects quiet ambient noise via calibrated RMS threshold and suppresses known Whisper silence hallucinations (`"You"`, `"Thank you."`, etc.).
- **Barge-in / Interrupt**: Pressing spacebar while LUCIA is speaking immediately stops audio playback and clears the queue so you can speak right away.
- **Sentence-by-Sentence Streaming**: Synthesizes and plays speech concurrently as LLM tokens arrive, minimizing the time before LUCIA begins speaking.
- **Decimal-Safe Speech**: Preserves decimal numbers (e.g. `23.5°C`, `0.1 mm`) without breaking them mid-sentence.
- **Automatic Fallback**: Attempts to run Whisper on NVIDIA CUDA GPU (`float16`), and automatically falls back to CPU (`int8`) with clear notification if GPU runtime/DLLs are not available.

### 3. Text Mode (Phase 1)
```bash
python main.py
```
Type your query and press Enter. Type `exit` or `quit` to exit.

---

## Try These Prompts

- *"What time is it?"*
- *"What's the weather in Bengaluru?"*
- *"Search the web for the latest news on SpaceX"*
- *"Open calculator"*
- *"Compare the weather in Delhi and Mumbai, which is hotter?"* (multi-tool)

---

## Architecture & Files

- `config.py`: Centralized configuration (`MODEL`, `SYSTEM_PROMPT`, `WAKE_*`, `WHISPER_*`, `PIPER_*`, `AUDIO_*`, `APP_WHITELIST`).
- `wakeword.py`: Wake word detection wrapper using `openWakeWord` with automatic model downloading, failure handling, and state resets.
- `main_wake.py`: Hands-free wake word entry point with end-of-speech detection and echo cooldown.
- `main_voice.py`: Push-to-talk voice mode entry point.
- `main.py`: Interactive text chat loop.
- `stt.py`: Speech-to-Text module using `faster-whisper` with automatic GPU-to-CPU fallback, silence RMS checks, and hallucination filters.
- `tts.py`: Text-to-Speech module using `piper-tts` on CPU, auto-voice downloader, decimal-safe sentence splitter, and streaming playback queue.
- `tools.py`: Actions registry (`get_current_time`, `get_weather`, `web_search`, `open_app`).
- `tests/`: Comprehensive test suite across config, history, tools, STT, TTS, and wake word detection.

---

## Configuration Settings (`config.py`)

All settings are configured in `config.py`:
- `WAKE_WORD_MODEL`: Pretrained wake word model (default: `"hey_jarvis"`).
- `WAKE_WORD_THRESHOLD`: Detection confidence threshold (default: `0.5`).
- `WAKE_END_OF_SPEECH_PAUSE`: Consecutive silence duration in seconds to stop recording (default: `1.2`).
- `WAKE_SPEECH_START_TIMEOUT`: Seconds to wait for speech after wake word before timing out (default: `3.5`).
- `WAKE_COOLDOWN_SECONDS`: Cooldown after TTS playback before re-arming wake detection (default: `1.0`).
- `MODEL`: LLM model name for Ollama (default: `"qwen2.5:7b"`).
- `WHISPER_MODEL_SIZE`: Whisper model size (default: `"small"`, fallback: `"base"`).
- `WHISPER_DEVICE`: Preferred compute device (default: `"cuda"`, auto-falls back to `"cpu"`).
- `PUSH_TO_TALK_KEY`: Push-to-talk hotkey for `main_voice.py` (default: `"space"`).
- `SILENCE_RMS_THRESHOLD`: Calibrated RMS silence threshold (default: `0.048`).
- `PIPER_VOICE_MODEL`: Path to Piper voice ONNX model (default: `"voices/en_US-lessac-medium.onnx"`).

