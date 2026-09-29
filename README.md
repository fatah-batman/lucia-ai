# LUCIA - Personal AI Assistant (100% Free & Local)

Inspired by JARVIS from Iron Man. Built in modular phases:
- **Phase 1**: Text assistant with Ollama tool calling, token streaming, and rolling history trimming.
- **Phase 2**: Local voice input (STT via `faster-whisper`) and output (TTS via `piper-tts`) with push-to-talk, barge-in, and decimal-safe sentence-streaming speech.

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

### 1. Voice Mode (Phase 2)
```bash
python main_voice.py
```
- **Push-to-Talk**: Hold **SPACEBAR** to speak, release to send.
- **Terminal Focus Guard**: On Windows, push-to-talk only activates when the terminal window has focus (configured via `REQUIRE_TERMINAL_FOCUS = True`), preventing accidental triggers while typing in other windows.
- **Barge-in / Interrupt**: Pressing spacebar while LUCIA is speaking immediately stops audio playback and clears the queue so you can speak right away.
- **Sentence-by-Sentence Streaming**: Synthesizes and plays speech concurrently as LLM tokens arrive, minimizing the time before LUCIA begins speaking.
- **Decimal-Safe Speech**: Preserves decimal numbers (e.g. `23.5°C`, `0.1 mm`) without breaking them mid-sentence.
- **Automatic Fallback**: Attempts to run Whisper on NVIDIA CUDA GPU (`float16`), and automatically falls back to CPU (`int8`) with clear notification if GPU runtime/DLLs are not available.
- **Piper Voice Model**: The default voice (`en_US-lessac-medium`, ~63 MB) is downloaded automatically to the `voices/` directory on first launch.

### 2. Text Mode (Phase 1)
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

- `config.py`: Centralized configuration (`MODEL`, `SYSTEM_PROMPT`, `WHISPER_*`, `PIPER_*`, `PUSH_TO_TALK_KEY`, `AUDIO_*`, `APP_WHITELIST`).
- `main_voice.py`: Voice mode entry point with push-to-talk loop, focus guard, STT transcription, and streaming TTS.
- `main.py`: Text mode entry point with interactive chat loop and streaming output.
- `stt.py`: Speech-to-Text module using `faster-whisper` with automatic GPU-to-CPU fallback and RMS silence filtering.
- `tts.py`: Text-to-Speech module using `piper-tts` on CPU, auto-voice downloader, decimal-safe sentence-boundary splitter, and background playback queue.
- `tools.py`: Actions registry (`get_current_time`, `get_weather`, `web_search`, `open_app`).
- `tests/`: Pytest suite covering configuration, history trimming, STT, TTS, and tool calling.

---

## Configuration Settings (`config.py`)

All settings are configured in `config.py`:
- `MODEL`: LLM model name for Ollama (default: `"qwen2.5:7b"`).
- `WHISPER_MODEL_SIZE`: Whisper model size (default: `"small"`, fallback: `"base"`).
- `WHISPER_DEVICE`: Preferred compute device (default: `"cuda"`, auto-falls back to `"cpu"`).
- `WHISPER_COMPUTE_TYPE`: Precision (default: `"float16"` for GPU, `"int8"` for CPU).
- `PUSH_TO_TALK_KEY`: Key to hold while speaking (default: `"space"`).
- `AUDIO_RMS_THRESHOLD`: Minimum audio energy threshold (default: `0.015`) to reject silence.
- `REQUIRE_TERMINAL_FOCUS`: Only capture push-to-talk when terminal is active (default: `True`).
- `PIPER_VOICE_MODEL`: Path to Piper voice ONNX model (default: `"voices/en_US-lessac-medium.onnx"`).
