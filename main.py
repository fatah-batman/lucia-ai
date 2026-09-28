"""
main.py - Phase 1 LUCIA: text in, text out, with tool calling.

Flow:
  you type -> LLM decides (answer OR call a tool) -> we run the tool ->
  feed result back -> LLM writes the final answer.

Run:  python main.py
"""

from typing import Callable, Optional
import sys
import ollama

from config import MODEL, SYSTEM_PROMPT, MAX_TOOL_ROUNDS, MAX_HISTORY_MESSAGES
from tools import TOOLS

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def trim_history(messages: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """Trim conversation history to a rolling window while preserving turn structure.

    Rules:
      - Always keep the system prompt at index 0 (if present).
      - If non-system messages <= max_messages, return intact.
      - Never split a tool exchange (tool calls + tool results).
      - Kept history always starts on a 'user' message.
    """
    if not messages:
        return []

    first = messages[0]
    has_system = (
        first.get("role") == "system"
        if isinstance(first, dict)
        else getattr(first, "role", None) == "system"
    )
    system_part = [first] if has_system else []
    non_system = messages[1:] if has_system else messages[:]

    if len(non_system) <= max_messages:
        return system_part + non_system

    cutoff = len(non_system) - max_messages
    start_idx = None
    for i in range(cutoff, len(non_system)):
        msg = non_system[i]
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "user":
            start_idx = i
            break

    if start_idx is None:
        for i in range(len(non_system) - 1, -1, -1):
            msg = non_system[i]
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "user":
                start_idx = i
                break

    if start_idx is None:
        return system_part

    return system_part + non_system[start_idx:]


def run_turn(messages: list, on_token: Optional[Callable[[str], None]] = None) -> str:
    """Send messages to the model, stream the reply, execute tool calls, return final text.

    Args:
        messages: Conversation history.
        on_token: Optional callback invoked for each streamed token chunk.
                  Defaults to printing to the console.
    """
    if on_token is None:
        def default_on_token(token: str) -> None:
            print(token, end="", flush=True)

        on_token = default_on_token

    messages[:] = trim_history(messages, MAX_HISTORY_MESSAGES)

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=list(TOOLS.values()),
                stream=True,
            )
        except Exception as e:
            err_msg = (
                f"I couldn't reach Ollama ({e}). "
                "Please make sure the Ollama application or service is running."
            )
            on_token(err_msg)
            return err_msg

        content = ""
        tool_calls = []

        for chunk in response:
            msg = chunk.message
            if msg.content:
                content += msg.content
                on_token(msg.content)
            if msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # No tool requested -> this is the final answer
        if not tool_calls:
            return content

        # Execute each requested tool and feed results back
        for call in tool_calls:
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

    stuck_msg = "I got stuck in a loop trying to do that. Try rephrasing?"
    on_token(stuck_msg)
    return stuck_msg


def main():
    print(f"LUCIA online (model: {MODEL}). Type 'exit' to quit.\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        snapshot = list(messages)
        messages.append({"role": "user", "content": user_input})

        first_token = [True]

        def handle_token(token: str) -> None:
            if first_token[0]:
                print("LUCIA: ", end="", flush=True)
                first_token[0] = False
            print(token, end="", flush=True)

        try:
            run_turn(messages, on_token=handle_token)
            print("\n")
        except KeyboardInterrupt:
            # Clean rollback of incomplete turn on Ctrl+C mid-reply
            messages[:] = snapshot
            print("\n")
            continue

    print("Shutting down. Goodbye.")


if __name__ == "__main__":
    main()
