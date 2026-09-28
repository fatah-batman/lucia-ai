# LUCIA - Phase 1 (text-only, 100% free & local)

## Setup

1. Install Ollama: https://ollama.com/download
2. Pull a model (one-time, ~4-5 GB):
   ```
   ollama pull qwen2.5:7b
   ```
   Low on RAM (<8 GB)? Use `qwen2.5:3b` or `llama3.2:3b` and change `MODEL` in main.py.
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

- `main.py`  - chat loop + tool-calling logic
- `tools.py` - the actions (add new tools here; just write a function with a docstring, then add it to `TOOLS`)

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
