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
