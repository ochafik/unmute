import pytest

from unmute.llm.llm_utils import preprocess_messages_for_llm, rechunk_to_words


class TestPreprocessMessagesForLLM:
    """Tests for preprocess_messages_for_llm function."""

    def test_tool_call_message_passthrough(self):
        """Test that tool call messages with None content pass through."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                    }
                ],
            },
        ]

        result = preprocess_messages_for_llm(messages)

        # Tool call message should be in output (system + user + tool call)
        assert len(result) == 3
        tool_call_msg = result[-1]
        assert tool_call_msg["role"] == "assistant"
        assert tool_call_msg["content"] is None
        assert "tool_calls" in tool_call_msg

    def test_tool_result_message_passthrough(self):
        """Test that tool result messages pass through."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": '{"temp": "15°C"}'},
        ]

        result = preprocess_messages_for_llm(messages)

        # Tool result message should be in output
        tool_result_msg = result[-1]
        assert tool_result_msg["role"] == "tool"
        assert tool_result_msg["tool_call_id"] == "call_123"

    def test_tool_messages_not_merged(self):
        """Test that tool messages are not merged with regular messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "Let me check."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        ]

        result = preprocess_messages_for_llm(messages)

        # Should have separate assistant messages, not merged
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0]["content"] == "Let me check."
        assert assistant_msgs[1]["content"] is None
        assert "tool_calls" in assistant_msgs[1]

    def test_full_tool_call_cycle(self):
        """Test a complete tool call cycle: user -> tool_call -> tool_result -> assistant."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": '{"temp": "15°C"}'},
            {"role": "assistant", "content": "The weather in Paris is 15°C."},
        ]

        result = preprocess_messages_for_llm(messages)

        # All messages should be preserved in correct order
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]


async def make_iterator(s: str):
    parts = s.split("|")
    for part in parts:
        yield part


@pytest.mark.asyncio
async def test_rechunk_to_words():
    test_strings = [
        "hel|lo| |w|orld",
        "hello world",
        "hello \nworld",
        "hello| |world",
        "hello| |world|.",
        "h|e|l|l|o| |\tw|o|r|l|d|.",
        "h|e|l|l|o\n| |w|o|r|l|d|.",
    ]

    for s in test_strings:
        parts = [x async for x in rechunk_to_words(make_iterator(s))]
        assert parts[0] == "hello"
        assert parts[1] == " world" or parts[1] == " world."

    async def f(s: str):
        x = [x async for x in rechunk_to_words(make_iterator(s))]
        print(x)
        return x

    assert await f("i am ok") == ["i", " am", " ok"]
    assert await f(" i am ok") == [" i", " am", " ok"]
    assert await f(" they are ok") == [" they", " are", " ok"]
    assert await f("  foo bar") == [" foo", " bar"]
    assert await f(" \t foo  bar") == [" foo", " bar"]
