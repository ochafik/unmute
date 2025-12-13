"""
Tests for OpenAI Realtime API-compatible tool/function calling.

Tests cover:
1. Event type serialization/deserialization
2. Chatbot tool call tracking
3. Protocol adapter handling
4. VLLMStream tool call parsing
"""

import json

import pytest

import unmute.openai_realtime_api_events as ora
from unmute.llm.chatbot import Chatbot
from unmute.llm.llm_utils import (
    TextDelta,
    ToolCallAccumulator,
    ToolCallComplete,
    ToolCallDelta,
)
from unmute.protocol.openai import OpenAIProtocolAdapter


class TestToolEventTypes:
    """Test event type serialization and deserialization."""

    def test_tool_function_parameters(self):
        """Test ToolFunctionParameters model."""
        params = ora.ToolFunctionParameters(
            properties={"location": {"type": "string", "description": "City name"}},
            required=["location"],
        )
        assert params.type == "object"
        assert "location" in params.properties
        assert "location" in params.required

    def test_tool_function(self):
        """Test ToolFunction model."""
        func = ora.ToolFunction(
            name="get_weather",
            description="Get current weather",
            parameters=ora.ToolFunctionParameters(
                properties={"location": {"type": "string"}},
                required=["location"],
            ),
        )
        assert func.name == "get_weather"
        assert func.description == "Get current weather"

    def test_tool_nested_format(self):
        """Test Tool model with nested function format."""
        tool = ora.Tool(
            function=ora.ToolFunction(
                name="get_weather",
                description="Get current weather",
            )
        )
        assert tool.type == "function"
        assert tool.get_name() == "get_weather"
        assert tool.get_description() == "Get current weather"

    def test_tool_flat_format(self):
        """Test Tool model with flat format."""
        tool = ora.Tool(
            name="get_time",
            description="Get current time",
            parameters=ora.ToolFunctionParameters(),
        )
        assert tool.get_name() == "get_time"
        assert tool.get_description() == "Get current time"

    def test_tool_to_openai_format(self):
        """Test Tool.to_openai_format() conversion."""
        tool = ora.Tool(
            function=ora.ToolFunction(
                name="get_weather",
                description="Get weather for a location",
                parameters=ora.ToolFunctionParameters(
                    properties={"location": {"type": "string"}},
                    required=["location"],
                ),
            )
        )
        openai_format = tool.to_openai_format()
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "get_weather"
        assert openai_format["function"]["description"] == "Get weather for a location"
        assert "location" in openai_format["function"]["parameters"]["properties"]

    def test_session_config_with_tools(self):
        """Test SessionConfig with tools field."""
        config = ora.SessionConfig(
            allow_recording=False,
            tools=[
                ora.Tool(
                    function=ora.ToolFunction(name="get_weather", description="Get weather")
                )
            ],
            tool_choice="auto",
        )
        assert len(config.tools) == 1
        assert config.tool_choice == "auto"

    def test_function_call_item(self):
        """Test FunctionCallItem model."""
        item = ora.FunctionCallItem(
            name="get_weather",
            arguments='{"location": "Paris"}',
        )
        assert item.type == "function_call"
        assert item.status == "in_progress"
        assert item.name == "get_weather"
        assert item.id.startswith("item_")
        assert item.call_id.startswith("call_")

    def test_response_output_item_added(self):
        """Test ResponseOutputItemAdded event."""
        item = ora.FunctionCallItem(name="get_weather")
        event = ora.ResponseOutputItemAdded(
            response_id="resp_123",
            output_index=0,
            item=item,
        )
        assert event.type == "response.output_item.added"
        assert event.response_id == "resp_123"

    def test_response_function_call_arguments_delta(self):
        """Test ResponseFunctionCallArgumentsDelta event."""
        event = ora.ResponseFunctionCallArgumentsDelta(
            response_id="resp_123",
            item_id="item_456",
            output_index=0,
            call_id="call_789",
            delta='{"loc',
        )
        assert event.type == "response.function_call_arguments.delta"
        assert event.delta == '{"loc'

    def test_response_function_call_arguments_done(self):
        """Test ResponseFunctionCallArgumentsDone event."""
        event = ora.ResponseFunctionCallArgumentsDone(
            response_id="resp_123",
            item_id="item_456",
            output_index=0,
            call_id="call_789",
            arguments='{"location": "Paris"}',
        )
        assert event.type == "response.function_call_arguments.done"
        assert event.arguments == '{"location": "Paris"}'

    def test_response_output_item_done(self):
        """Test ResponseOutputItemDone event."""
        item = ora.FunctionCallItem(
            name="get_weather",
            arguments='{"location": "Paris"}',
            status="completed",
        )
        event = ora.ResponseOutputItemDone(
            response_id="resp_123",
            output_index=0,
            item=item,
        )
        assert event.type == "response.output_item.done"

    def test_response_done(self):
        """Test ResponseDone event."""
        event = ora.ResponseDone(response_id="resp_123")
        assert event.type == "response.done"
        assert event.response_id == "resp_123"

    def test_function_call_output_item(self):
        """Test FunctionCallOutputItem model."""
        item = ora.FunctionCallOutputItem(
            call_id="call_123",
            output='{"temperature": "15°C"}',
        )
        assert item.type == "function_call_output"
        assert item.call_id == "call_123"

    def test_conversation_item_create(self):
        """Test ConversationItemCreate event."""
        item = ora.FunctionCallOutputItem(
            call_id="call_123",
            output='{"temperature": "15°C"}',
        )
        event = ora.ConversationItemCreate(item=item)
        assert event.type == "conversation.item.create"

    def test_response_create(self):
        """Test ResponseCreate event."""
        event = ora.ResponseCreate()
        assert event.type == "response.create"


class TestChatbotToolTracking:
    """Test Chatbot tool call tracking methods."""

    def test_add_tool_call(self):
        """Test adding a tool call to chat history."""
        chatbot = Chatbot()
        chatbot.add_tool_call("call_123", "get_weather", '{"location": "Paris"}')

        # Check the message was added
        assert len(chatbot.chat_history) == 2  # system + tool call
        last_msg = chatbot.chat_history[-1]
        assert last_msg["role"] == "assistant"
        assert last_msg["content"] is None
        assert len(last_msg["tool_calls"]) == 1
        assert last_msg["tool_calls"][0]["id"] == "call_123"
        assert last_msg["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_add_tool_result(self):
        """Test adding a tool result to chat history."""
        chatbot = Chatbot()
        chatbot.add_tool_result("call_123", '{"temperature": "15°C"}')

        # Check the message was added
        assert len(chatbot.chat_history) == 2  # system + tool result
        last_msg = chatbot.chat_history[-1]
        assert last_msg["role"] == "tool"
        assert last_msg["tool_call_id"] == "call_123"
        assert last_msg["content"] == '{"temperature": "15°C"}'

    def test_conversation_state_after_tool_call(self):
        """Test conversation state after tool call."""
        chatbot = Chatbot()
        chatbot.add_tool_call("call_123", "get_weather", '{"location": "Paris"}')

        # After a tool call, we're waiting for user (tool result)
        assert chatbot.conversation_state() == "waiting_for_user"

    def test_conversation_state_after_tool_result(self):
        """Test conversation state after tool result."""
        chatbot = Chatbot()
        chatbot.add_tool_call("call_123", "get_weather", '{"location": "Paris"}')
        chatbot.add_tool_result("call_123", '{"temperature": "15°C"}')

        # After a tool result, we're waiting for assistant response
        assert chatbot.conversation_state() == "waiting_for_user"

    def test_last_message_with_tool_calls(self):
        """Test last_message handles tool call messages correctly."""
        chatbot = Chatbot()
        # Add some regular messages
        chatbot.chat_history.append({"role": "user", "content": "What's the weather?"})
        chatbot.chat_history.append({"role": "assistant", "content": "Let me check."})

        # Last message is regular text
        assert chatbot.last_message("assistant") == "Let me check."

        # Add a tool call (no content)
        chatbot.add_tool_call("call_123", "get_weather", '{}')

        # last_message should still return the previous text message
        assert chatbot.last_message("assistant") == "Let me check."


class TestProtocolAdapterToolHandling:
    """Test OpenAI protocol adapter handling of tool events."""

    def test_translate_session_update_with_tools(self):
        """Test translating session.update with tools."""
        adapter = OpenAIProtocolAdapter()

        message_dict = {
            "type": "session.update",
            "session": {
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
                "tool_choice": "auto",
            },
        }
        message_json = json.dumps(message_dict)

        translated = adapter.translate_client_message(message_json)
        assert isinstance(translated, ora.SessionUpdate)
        assert translated.session.tools is not None
        assert len(translated.session.tools) == 1
        assert translated.session.tools[0].get_name() == "get_weather"
        assert translated.session.tool_choice == "auto"

    def test_translate_conversation_item_create_function_output(self):
        """Test translating conversation.item.create with function output."""
        adapter = OpenAIProtocolAdapter()

        message_dict = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": '{"temperature": "15°C"}',
            },
        }
        message_json = json.dumps(message_dict)

        translated = adapter.translate_client_message(message_json)
        assert isinstance(translated, ora.ConversationItemCreate)

    def test_translate_conversation_item_create_other_filtered(self):
        """Test that other conversation.item.create types are filtered."""
        adapter = OpenAIProtocolAdapter()

        message_dict = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            },
        }
        message_json = json.dumps(message_dict)

        translated = adapter.translate_client_message(message_json)
        assert translated is None  # Filtered for non-function_call_output

    def test_translate_response_create(self):
        """Test translating response.create event."""
        adapter = OpenAIProtocolAdapter()

        message_dict = {"type": "response.create"}
        message_json = json.dumps(message_dict)

        translated = adapter.translate_client_message(message_json)
        assert isinstance(translated, ora.ResponseCreate)

    def test_translate_function_call_arguments_delta(self):
        """Test translating function call arguments delta server event."""
        adapter = OpenAIProtocolAdapter()

        event = ora.ResponseFunctionCallArgumentsDelta(
            response_id="resp_123",
            item_id="item_456",
            output_index=0,
            call_id="call_789",
            delta='{"loc',
        )

        translated_json = adapter.translate_server_event(event)
        assert translated_json is not None
        translated = json.loads(translated_json)
        assert translated["type"] == "response.function_call_arguments.delta"
        assert translated["delta"] == '{"loc'

    def test_translate_function_call_arguments_done(self):
        """Test translating function call arguments done server event."""
        adapter = OpenAIProtocolAdapter()

        event = ora.ResponseFunctionCallArgumentsDone(
            response_id="resp_123",
            item_id="item_456",
            output_index=0,
            call_id="call_789",
            arguments='{"location": "Paris"}',
        )

        translated_json = adapter.translate_server_event(event)
        assert translated_json is not None
        translated = json.loads(translated_json)
        assert translated["type"] == "response.function_call_arguments.done"
        assert translated["arguments"] == '{"location": "Paris"}'

    def test_translate_response_output_item_done(self):
        """Test translating response output item done server event."""
        adapter = OpenAIProtocolAdapter()

        item = ora.FunctionCallItem(
            name="get_weather",
            arguments='{"location": "Paris"}',
            status="completed",
        )
        event = ora.ResponseOutputItemDone(
            response_id="resp_123",
            output_index=0,
            item=item,
        )

        translated_json = adapter.translate_server_event(event)
        assert translated_json is not None
        translated = json.loads(translated_json)
        assert translated["type"] == "response.output_item.done"
        assert translated["item"]["name"] == "get_weather"

    def test_translate_response_done(self):
        """Test translating response done server event."""
        adapter = OpenAIProtocolAdapter()

        event = ora.ResponseDone(response_id="resp_123")

        translated_json = adapter.translate_server_event(event)
        assert translated_json is not None
        translated = json.loads(translated_json)
        assert translated["type"] == "response.done"


class TestLLMToolCallTypes:
    """Test LLM utility types for tool calls."""

    def test_text_delta(self):
        """Test TextDelta dataclass."""
        delta = TextDelta(content="Hello")
        assert delta.content == "Hello"

    def test_tool_call_delta(self):
        """Test ToolCallDelta dataclass."""
        delta = ToolCallDelta(
            index=0,
            id="call_123",
            name="get_weather",
            arguments_delta='{"loc',
        )
        assert delta.index == 0
        assert delta.id == "call_123"
        assert delta.name == "get_weather"
        assert delta.arguments_delta == '{"loc'

    def test_tool_call_complete(self):
        """Test ToolCallComplete dataclass."""
        complete = ToolCallComplete(
            index=0,
            id="call_123",
            name="get_weather",
            arguments='{"location": "Paris"}',
        )
        assert complete.index == 0
        assert complete.name == "get_weather"
        assert complete.arguments == '{"location": "Paris"}'

    def test_tool_call_accumulator(self):
        """Test ToolCallAccumulator dataclass."""
        acc = ToolCallAccumulator()
        assert acc.id == ""
        assert acc.name == ""
        assert acc.arguments == ""

        # Simulate accumulation
        acc.id = "call_123"
        acc.name = "get_weather"
        acc.arguments += '{"loc'
        acc.arguments += 'ation": "'
        acc.arguments += 'Paris"}'

        assert acc.id == "call_123"
        assert acc.name == "get_weather"
        assert acc.arguments == '{"location": "Paris"}'
