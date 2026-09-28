"""tests/test_history.py - Unit tests for conversation history trimming."""

from main import trim_history


def test_trim_history_under_limit():
    """History under the limit should remain unchanged."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = trim_history(messages, max_messages=10)
    assert result == messages


def test_trim_history_preserves_system_prompt():
    """System prompt must always be preserved at index 0."""
    messages = [{"role": "system", "content": "system prompt"}]
    for i in range(15):
        messages.append({"role": "user", "content": f"user {i}"})
        messages.append({"role": "assistant", "content": f"assistant {i}"})

    result = trim_history(messages, max_messages=6)
    assert len(result) > 1
    assert result[0] == {"role": "system", "content": "system prompt"}


def test_trim_history_starts_on_user_message():
    """The kept non-system messages must always start on a user message."""
    messages = [{"role": "system", "content": "system prompt"}]
    for i in range(10):
        messages.append({"role": "user", "content": f"user {i}"})
        messages.append({"role": "assistant", "content": f"assistant {i}"})

    # Total 20 non-system messages. Trim to 5.
    result = trim_history(messages, max_messages=5)
    # result[0] is system, result[1] must be user
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert len(result[1:]) <= 5


def test_trim_history_never_splits_tool_exchange():
    """Tool calls and their results must be kept or dropped together."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "check weather"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Bengaluru"}}}],
        },
        {"role": "tool", "tool_name": "get_weather", "content": "24°C"},
        {"role": "assistant", "content": "It is 24°C in Bengaluru."},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "you are welcome"},
    ]
    # Non-system messages count = 6.
    # If max_messages = 4, cutoff is 6 - 4 = 2 (index 2 in non_system is the 'tool' result message).
    # If naively sliced, it would split the assistant tool call from the tool result.
    result = trim_history(messages, max_messages=4)

    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    # Should drop the tool exchange completely rather than splitting it, keeping 'thanks' turn
    assert result[1]["content"] == "thanks"
    assert result[2]["content"] == "you are welcome"
    assert len(result[1:]) <= 4


def test_trim_history_empty_and_no_system():
    """Empty history or history without system prompt should be handled gracefully."""
    assert trim_history([]) == []

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = trim_history(messages, max_messages=10)
    assert result == messages
