# LUCIA - Phase 1 (text-only, 100% free & local)

## Setup

1. Install Ollama: https://ollama.com/download
2. Pull a model (one-time, ~4-5 GB):
   ```
   ollama pull qwen2.5:7b
   ```
   Low on RAM (<8 GB)? Use `qwen2.5:3b` or `llama3.2:3b` and change `MODEL` in `config.py`.
3. Create and activate a virtual environment (Python 3.10+):
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
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run unit tests:
   ```bash
   pytest
   ```
6. Run LUCIA:
   ```bash
   python main.py
   ```

## Try these

- "What time is it?"
- "What's the weather in Bengaluru?"
- "Search the web for the latest news on SpaceX"
- "Open calculator"
- "Weather in Delhi and Mumbai, which is hotter?" (multi-tool)

## Files

- `config.py` - central configuration (`MODEL`, `SYSTEM_PROMPT`, `MAX_TOOL_ROUNDS`, `MAX_HISTORY_MESSAGES`, `APP_WHITELIST`)
- `main.py`   - chat loop, streaming response handling (`stream=True`, pluggable `on_token`), tool-calling logic, and history trimming
- `tools.py`  - the actions (add new tools here; write a function with a docstring, then register it in `TOOLS`)
- `tests/`    - pytest test suite for tools, central config, and history trimming

## Configuration

All settings are centralized in `config.py`. To switch models (e.g. to `qwen2.5:14b` or a smaller `qwen2.5:3b`), change `MODEL` in `config.py`:

```python
MODEL = "qwen2.5:14b"
```

## Adding a tool

```python
def my_tool(arg: str) -> str:
    """One clear sentence on what this does.

    Args:
        arg: what this argument means.
    """
    return "result"

TOOLS["my_tool"] = my_tool
```
