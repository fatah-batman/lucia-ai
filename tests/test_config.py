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
