"""config.py - Central configuration for LUCIA."""

# LLM model to run locally via Ollama
MODEL = "qwen2.5:7b"

# System personality prompt
SYSTEM_PROMPT = (
    "You are LUCIA, a witty, concise personal assistant. "
    "Use tools when they help (weather, time, web search, opening apps). "
    "Never invent tool results. If a tool fails, say so plainly. "
    "Keep answers short and conversational."
)

# Safety caps
MAX_TOOL_ROUNDS = 5  # Cap to prevent infinite tool loops
MAX_HISTORY_MESSAGES = 20  # Rolling window size for conversation history

# Whitelist of apps LUCIA is permitted to open per OS
APP_WHITELIST = {
    "Windows": {
        "notepad": "notepad",
        "calculator": "calc",
        "calc": "calc",
        "chrome": "chrome",
        "vscode": "code",
        "code": "code",
        "explorer": "explorer",
    },
    "Darwin": {  # macOS
        "safari": "Safari",
        "calculator": "Calculator",
        "chrome": "Google Chrome",
        "vscode": "Visual Studio Code",
        "terminal": "Terminal",
    },
    "Linux": {
        "calculator": "gnome-calculator",
        "chrome": "google-chrome",
        "vscode": "code",
        "terminal": "gnome-terminal",
        "files": "nautilus",
    },
}

# Voice & Audio Configuration (Phase 2)
WHISPER_MODEL_SIZE = "small"          # Whisper model size ("small" default, "base" fallback)
WHISPER_DEVICE = "cuda"               # Preferred device: "cuda" on GPU, auto-falls back to "cpu"
WHISPER_COMPUTE_TYPE = "float16"      # Compute precision: "float16" on GPU, "int8" on CPU fallback

PIPER_VOICE_MODEL = "voices/en_US-lessac-medium.onnx"
PIPER_VOICE_CONFIG = "voices/en_US-lessac-medium.onnx.json"
PIPER_VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

PUSH_TO_TALK_KEY = "space"            # Push-to-talk key to hold while speaking
AUDIO_SAMPLE_RATE = 16000             # Standard sampling rate for Whisper STT (16 kHz)
AUDIO_CHANNELS = 1                    # Mono recording
AUDIO_MIN_DURATION_SECONDS = 0.3      # Minimum press duration in seconds to trigger processing
SILENCE_RMS_THRESHOLD = 0.035          # Calibrated RMS threshold for reliable speech capture
AUDIO_RMS_THRESHOLD = SILENCE_RMS_THRESHOLD  # Backward compatibility alias
REQUIRE_TERMINAL_FOCUS = False        # Set to False so it works reliably in IDE terminals and Windows Terminal


# Wake Word & Hands-Free Configuration (Phase 3)
WAKE_WORD_MODEL = "hey_jarvis"            # Pretrained openWakeWord model
WAKE_WORD_THRESHOLD = 0.5                # Activation score threshold (0.0 to 1.0)
WAKE_SPEECH_START_TIMEOUT = 3.5          # Seconds to wait for speech to start after wake word
WAKE_END_OF_SPEECH_PAUSE = 1.2           # Consecutive seconds of silence to detect end of speech
WAKE_SILENCE_RMS_THRESHOLD = SILENCE_RMS_THRESHOLD  # RMS silence threshold for hands-free speech detection
WAKE_PRE_ROLL_SECONDS = 0.5              # Pre-roll audio buffer to prevent clipping first syllable
WAKE_MAX_RECORDING_SECONDS = 15.0        # Max safety recording limit
WAKE_COOLDOWN_SECONDS = 1.0              # Post-TTS playback pause before resuming wake word listening

# Known Whisper hallucinations on quiet ambient noise/silence
WHISPER_HALLUCINATION_PHRASES = [
    "you",
    "thank you",
    "thanks for watching",
    "bye",
    "bye.",
    "thank you very much",
    "thanks",
    "so",
    "yeah",
    "subscribe",
    "watching",
]

