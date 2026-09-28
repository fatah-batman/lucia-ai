# Antigravity Prompt: LUCIA Phase 1

Copy everything below the line into Antigravity's agent (Manager Surface), with this `lucia` folder open as the workspace.

---

You are helping me build **LUCIA**, a personal AI voice assistant inspired by JARVIS from Iron Man. We are building it in phases. This is **Phase 1: a text-only assistant with tool calling**. Voice, wake word, and a web UI come in later phases, so do NOT add them now.

## Hard constraints
- **100% free and local.** No paid APIs, no API keys. The LLM runs locally through **Ollama** (model: `qwen2.5:7b`).
- Language: **Python 3.10+**. Keep dependencies to `ollama`, `requests`, and `ddgs`.
- Keep the current architecture: `main.py` (chat loop and tool-calling logic) and `tools.py` (the actions and the `TOOLS` registry). Do not restructure into frameworks like LangChain.
- Every tool must be a plain Python function with type hints and a clear docstring, because Ollama builds the tool schema from those docstrings.
- `open_app` must stay whitelist-only. Never let the model run arbitrary shell commands.

## Existing files
- `main.py`: chat loop, calls `ollama.chat` with tools, executes tool calls, feeds results back, capped at 5 tool rounds.
- `tools.py`: `get_current_time`, `get_weather` (Open-Meteo), `web_search` (DuckDuckGo via `ddgs`), `open_app` (whitelist).
- `requirements.txt`, `README.md`.

## Your tasks
1. **Plan first.** Produce an implementation plan artifact and wait for my approval before changing code.
2. **Set up the environment:** create a virtual environment, install `requirements.txt`, and check that Ollama is installed and `qwen2.5:7b` is pulled. If not, tell me the exact commands to run instead of guessing.
3. **Run LUCIA and test each tool** with these prompts:
   - "What time is it?"
   - "What's the weather in Bengaluru?"
   - "Search the web for the latest news on SpaceX"
   - "Open calculator"
   - "Compare the weather in Delhi and Mumbai" (multi-tool)
4. **Fix bugs you find**, especially Ollama API mismatches (for example `tool_calls` handling or the `tool_name` field in tool messages), `ddgs` import errors, and Windows-specific `open_app` behavior. Explain each fix briefly.
5. **Improve robustness without adding features:** graceful handling when Ollama isn't running, network timeouts in tools, and clean exit on Ctrl+C.
6. **Add a `tests/` folder** with simple pytest tests for the tools that don't need the LLM (`get_current_time`, and `get_weather` / `web_search` with mocked network calls).
7. **Update `README.md`** with anything that changed.

## Rules for working
- Make small, reviewable changes and summarize what you changed after each step.
- If something is ambiguous, ask me instead of choosing silently.
- Do not add features outside this list.

## Definition of done
`python main.py` starts LUCIA, all five test prompts above give sensible answers with the correct tool calls visible in the console, and the tests pass.
