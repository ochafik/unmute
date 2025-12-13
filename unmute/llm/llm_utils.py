import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache
from typing import Any, AsyncIterator, Protocol, cast

from mistralai import Mistral
from openai import AsyncOpenAI, OpenAI

from unmute.kyutai_constants import LLM_SERVER

from ..kyutai_constants import KYUTAI_LLM_API_KEY, KYUTAI_LLM_MODEL


# =============================================================================
# LLM Response Types (for tool calling support)
# =============================================================================


@dataclass
class TextDelta:
    """A text content delta from the LLM."""

    content: str


@dataclass
class ToolCallDelta:
    """A streaming delta for a tool call's arguments."""

    index: int
    id: str
    name: str  # May be empty for subsequent deltas
    arguments_delta: str


@dataclass
class ToolCallComplete:
    """A complete tool call from the LLM."""

    index: int
    id: str
    name: str
    arguments: str


# Type for what chat_completion_with_tools yields
LLMStreamItem = TextDelta | ToolCallDelta | ToolCallComplete


@dataclass
class ToolCallAccumulator:
    """Accumulates streaming tool call data."""

    id: str = ""
    name: str = ""
    arguments: str = ""

INTERRUPTION_CHAR = "—"  # em-dash
USER_SILENCE_MARKER = "..."


def preprocess_messages_for_llm(
    chat_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    for message in chat_history:
        message = deepcopy(message)

        # Handle tool call messages (assistant messages with tool_calls but no content)
        # and tool result messages (role="tool") - pass them through without modification
        if message.get("tool_calls") is not None or message.get("role") == "tool":
            output.append(message)
            continue

        content = message.get("content")

        # Skip messages with None content (shouldn't happen for normal messages)
        if content is None:
            continue

        # Sometimes, an interruption happens before the LLM can say anything at all.
        # In that case, we're left with a message with only INTERRUPTION_CHAR.
        # Simplify by removing.
        if content.replace(INTERRUPTION_CHAR, "") == "":
            continue

        # If the llm was interrupted we don't want to insert the INTERRUPTION_CHAR
        # into the context, otherwise the LLM might want to repeat it.
        message["content"] = content.strip().removesuffix(INTERRUPTION_CHAR)

        # Merge consecutive messages with the same role (but not tool-related messages)
        if (
            output
            and message["role"] == output[-1].get("role")
            and output[-1].get("tool_calls") is None
            and output[-1].get("role") != "tool"
        ):
            output[-1]["content"] += " " + message["content"]
        else:
            output.append(message)

    def role_at(index: int) -> str | None:
        if index >= len(output):
            return None
        return output[index]["role"]

    if role_at(0) == "system" and role_at(1) in [None, "assistant"]:
        # Some LLMs, like Gemma, get confused if the assistant message goes before user
        # messages, so add a dummy user message.
        output = [output[0]] + [{"role": "user", "content": "Hello."}] + output[1:]

    for message in chat_history:
        content = message.get("content")
        if (
            message["role"] == "user"
            and content is not None
            and content.startswith(USER_SILENCE_MARKER)
            and content != USER_SILENCE_MARKER
        ):
            # This happens when the user is silent but then starts talking again after
            # the silence marker was inserted but before the LLM could respond.
            # There are special instructions in the system prompt about how to handle
            # the silence marker, so remove the marker from the message to not confuse
            # the LLM
            message["content"] = content[len(USER_SILENCE_MARKER) :]

    return output


async def rechunk_to_words(iterator: AsyncIterator[str]) -> AsyncIterator[str]:
    """Rechunk the stream of text to whole words.

    Otherwise the TTS doesn't know where word boundaries are and will mispronounce
    split words.

    The spaces will be included with the next word, so "foo bar baz" will be split into
    "foo", " bar", " baz".
    Multiple space-like characters will be merged to a single space.
    """
    buffer = ""
    space_re = re.compile(r"\s+")
    prefix = ""
    async for delta in iterator:
        buffer = buffer + delta
        while True:
            match = space_re.search(buffer)
            if match is None:
                break
            chunk = buffer[: match.start()]
            buffer = buffer[match.end() :]
            if chunk != "":
                yield prefix + chunk
            prefix = " "

    if buffer != "":
        yield prefix + buffer


class LLMStream(Protocol):
    async def chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Get a chat completion from the LLM."""
        ...


class MistralStream:
    def __init__(self):
        self.current_message_index = 0
        self.mistral = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    async def chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        event_stream = await self.mistral.chat.stream_async(
            model="mistral-large-latest",
            messages=cast(Any, messages),  # It's too annoying to type this properly
            temperature=1.0,
        )

        async for event in event_stream:
            delta = event.data.choices[0].delta.content
            assert isinstance(delta, str)  # make Pyright happy
            yield delta


def get_openai_client(
    server_url: str = LLM_SERVER, api_key: str | None = KYUTAI_LLM_API_KEY
) -> AsyncOpenAI:
    # AsyncOpenAI() will complain if the API key is not set, so set a dummy string if it's None.
    # This still makes sense when using vLLM because it doesn't care about the API key.
    return AsyncOpenAI(api_key=api_key or "EMPTY", base_url=server_url + "/v1")


@cache
def autoselect_model() -> str:
    if KYUTAI_LLM_MODEL is not None:
        return KYUTAI_LLM_MODEL
    openai_client = get_openai_client()
    # OpenAI() will complain if the API key is not set, so set a dummy string if it's None.
    # This still makes sense when using vLLM because it doesn't care about the API key.
    client_sync = OpenAI(
        api_key=openai_client.api_key or "EMPTY", base_url=openai_client.base_url
    )
    models = client_sync.models.list()
    if len(models.data) != 1:
        raise ValueError("There are multiple models available. Please specify one.")
    return models.data[0].id


class VLLMStream:
    def __init__(
        self,
        client: AsyncOpenAI,
        temperature: float = 1.0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ):
        """
        If `model` is None, it will look at the available models, and if there is only
        one model, it will use that one. Otherwise, it will raise.

        Args:
            client: AsyncOpenAI client
            temperature: Sampling temperature
            tools: List of tool definitions in OpenAI format
            tool_choice: Tool choice mode ("auto", "none", "required", or specific tool)
        """
        self.client = client
        self.model = autoselect_model()
        self.temperature = temperature
        self.tools = tools
        self.tool_choice = tool_choice

    async def chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Legacy method for text-only completion (backward compatible)."""
        async for item in self.chat_completion_with_tools(messages):
            if isinstance(item, TextDelta):
                yield item.content

    async def chat_completion_with_tools(
        self, messages: list[dict[str, Any]]
    ) -> AsyncIterator[LLMStreamItem]:
        """
        Stream chat completion with support for tool calls.

        Yields:
            TextDelta: For text content
            ToolCallDelta: For streaming tool call arguments
            ToolCallComplete: When a tool call is finished
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": cast(Any, messages),
            "stream": True,
            "temperature": self.temperature,
        }

        # Add tools if configured
        if self.tools:
            params["tools"] = self.tools
        if self.tool_choice is not None:
            params["tool_choice"] = self.tool_choice

        stream = await self.client.chat.completions.create(**params)

        # Track tool calls being accumulated
        tool_call_accumulators: dict[int, ToolCallAccumulator] = {}

        async with stream:
            async for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Handle text content
                if delta.content:
                    yield TextDelta(content=delta.content)

                # Handle tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index

                        # Initialize accumulator for new tool calls
                        if idx not in tool_call_accumulators:
                            tool_call_accumulators[idx] = ToolCallAccumulator()

                        acc = tool_call_accumulators[idx]

                        # Update accumulator with new data
                        if tc.id:
                            acc.id = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc.name = tc.function.name
                            if tc.function.arguments:
                                acc.arguments += tc.function.arguments

                                # Emit delta for streaming arguments
                                yield ToolCallDelta(
                                    index=idx,
                                    id=acc.id,
                                    name=acc.name,
                                    arguments_delta=tc.function.arguments,
                                )

                # Check for finish reason indicating tool calls complete
                if choice.finish_reason == "tool_calls":
                    # Emit complete tool calls
                    for idx, acc in sorted(tool_call_accumulators.items()):
                        yield ToolCallComplete(
                            index=idx,
                            id=acc.id,
                            name=acc.name,
                            arguments=acc.arguments,
                        )
                    tool_call_accumulators.clear()
