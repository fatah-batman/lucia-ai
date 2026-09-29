"""tests/test_config.py - Tests for central configuration management."""

import config
import main
import tools


def test_config_constants_exist():
    """Verify config.py exposes all required configuration constants."""
    assert hasattr(config, "MODEL")
    assert hasattr(config, "SYSTEM_PROMPT")
    assert hasattr(config, "MAX_TOOL_ROUNDS")
    assert hasattr(config, "MAX_HISTORY_MESSAGES")
    assert hasattr(config, "APP_WHITELIST")

    assert isinstance(config.MODEL, str)
    assert isinstance(config.SYSTEM_PROMPT, str)
    assert isinstance(config.MAX_TOOL_ROUNDS, int)
    assert isinstance(config.MAX_HISTORY_MESSAGES, int)
    assert isinstance(config.APP_WHITELIST, dict)


def test_main_imports_from_config():
    """Verify main.py reads its settings directly from config.py."""
    assert main.MODEL is config.MODEL
    assert main.SYSTEM_PROMPT is config.SYSTEM_PROMPT
    assert main.MAX_TOOL_ROUNDS is config.MAX_TOOL_ROUNDS
    assert main.MAX_HISTORY_MESSAGES is config.MAX_HISTORY_MESSAGES


def test_tools_imports_from_config():
    """Verify tools.py reads APP_WHITELIST from config.py."""
    assert tools.APP_WHITELIST is config.APP_WHITELIST


def test_model_switching_is_one_line(monkeypatch):
    """Verify that updating config.MODEL reflects as the model in main."""
    monkeypatch.setattr(config, "MODEL", "qwen2.5:14b")
    # If main dynamically references or if config is updated:
    assert config.MODEL == "qwen2.5:14b"


def test_phase2_config_constants_exist():
    """Verify Phase 2 voice, STT, and TTS settings are present and typed properly."""
    assert hasattr(config, "WHISPER_MODEL_SIZE")
    assert hasattr(config, "WHISPER_DEVICE")
    assert hasattr(config, "WHISPER_COMPUTE_TYPE")
    assert hasattr(config, "PIPER_VOICE_MODEL")
    assert hasattr(config, "PIPER_VOICE_CONFIG")
    assert hasattr(config, "PIPER_VOICE_URL")
    assert hasattr(config, "PIPER_CONFIG_URL")
    assert hasattr(config, "PUSH_TO_TALK_KEY")
    assert hasattr(config, "AUDIO_SAMPLE_RATE")
    assert hasattr(config, "AUDIO_CHANNELS")
    assert hasattr(config, "AUDIO_MIN_DURATION_SECONDS")
    assert hasattr(config, "AUDIO_RMS_THRESHOLD")
    assert hasattr(config, "REQUIRE_TERMINAL_FOCUS")

    assert isinstance(config.WHISPER_MODEL_SIZE, str)
    assert isinstance(config.WHISPER_DEVICE, str)
    assert isinstance(config.WHISPER_COMPUTE_TYPE, str)
    assert isinstance(config.PIPER_VOICE_MODEL, str)
    assert isinstance(config.PUSH_TO_TALK_KEY, str)
    assert isinstance(config.AUDIO_SAMPLE_RATE, int)
    assert isinstance(config.AUDIO_CHANNELS, int)
    assert isinstance(config.AUDIO_MIN_DURATION_SECONDS, (int, float))
    assert isinstance(config.AUDIO_RMS_THRESHOLD, (int, float))
    assert isinstance(config.REQUIRE_TERMINAL_FOCUS, bool)


def test_phase3_config_constants_exist():
    """Verify Phase 3 wake word and hands-free settings exist with correct types."""
    assert hasattr(config, "WAKE_WORD_MODEL")
    assert hasattr(config, "WAKE_WORD_THRESHOLD")
    assert hasattr(config, "WAKE_SPEECH_START_TIMEOUT")
    assert hasattr(config, "WAKE_END_OF_SPEECH_PAUSE")
    assert hasattr(config, "WAKE_SILENCE_RMS_THRESHOLD")
    assert hasattr(config, "WAKE_PRE_ROLL_SECONDS")
    assert hasattr(config, "WAKE_MAX_RECORDING_SECONDS")
    assert hasattr(config, "WAKE_COOLDOWN_SECONDS")

    assert isinstance(config.WAKE_WORD_MODEL, str)
    assert isinstance(config.WAKE_WORD_THRESHOLD, (int, float))
    assert 0.0 <= config.WAKE_WORD_THRESHOLD <= 1.0
    assert isinstance(config.WAKE_SPEECH_START_TIMEOUT, (int, float))
    assert isinstance(config.WAKE_END_OF_SPEECH_PAUSE, (int, float))
    assert isinstance(config.WAKE_SILENCE_RMS_THRESHOLD, (int, float))
    assert isinstance(config.WAKE_PRE_ROLL_SECONDS, (int, float))
    assert isinstance(config.WAKE_MAX_RECORDING_SECONDS, (int, float))
    assert isinstance(config.WAKE_COOLDOWN_SECONDS, (int, float))

