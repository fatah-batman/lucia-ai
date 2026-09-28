"""
main.py - Phase 1 LUCIA: text in, text out, with tool calling.

Flow:
  you type -> LLM decides (answer OR call a tool) -> we run the tool ->
  feed result back -> LLM writes the final answer.

Run:  python main.py
"""

import sys
import ollama

from tools import TOOLS

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL = "qwen2.5:7b"  # good tool-calling for its size; alternative: "llama3.1:8b"
MAX_TOOL_ROUNDS = 5   # safety cap so the agent can't loop forever

SYSTEM_PROMPT = (
    "You are LUCIA, a witty, concise personal assistant. "
    "Use tools when they help (weather, time, web search, opening apps). "
    "Never invent tool results. If a tool fails, say so plainly. "
    "Keep answers short and conversational."
)


def run_turn(messages: list) -> str:
    """Send messages to the model, execute any tool calls, return final text."""
    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=list(TOOLS.values()),  # Ollama builds schemas from docstrings
            )
        except Exception as e:
            return (
                f"I couldn't reach Ollama ({e}). "
                "Please make sure the Ollama application or service is running."
            )

        msg = response.message
        messages.append(msg)

        # No tool requested -> this is the final answer
        if not msg.tool_calls:
            return msg.content or ""

        # Execute each requested tool and feed results back
        for call in msg.tool_calls:
            if hasattr(call, "function"):
                name = call.function.name
                args = call.function.arguments or {}
            else:
                name = call.get("function", {}).get("name")
                args = call.get("function", {}).get("arguments", {})

            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            print(f"  [tool] {name}({args})")

            fn = TOOLS.get(name)
            if fn is None:
                result = f"Unknown tool '{name}'."
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = f"Tool error: {e}"

            messages.append(
                {"role": "tool", "tool_name": name, "content": str(result)}
            )

    return "I got stuck in a loop trying to do that. Try rephrasing?"


def main():
    print(f"LUCIA online (model: {MODEL}). Type 'exit' to quit.\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            messages.append({"role": "user", "content": user_input})
            reply = run_turn(messages)
            print(f"LUCIA: {reply}\n")
        except (EOFError, KeyboardInterrupt):
            print()
            break

    print("Shutting down. Goodbye.")


if __name__ == "__main__":
    main()
