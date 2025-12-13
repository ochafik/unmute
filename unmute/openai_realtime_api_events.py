"""See OpenAI's docs: https://platform.openai.com/docs/api-reference/realtime

https://platform.openai.com/docs/api-reference/realtime-client-events
https://platform.openai.com/docs/api-reference/realtime-server-events
"""

import random
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, model_validator

from unmute.llm.system_prompt import Instructions

T = TypeVar("T", bound=str)


def random_id(prefix: str) -> str:
    """e.g. event_BJhGUIswO2u7vA2Cxw3Jy"""
    n_characters = 21
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return prefix + "_" + "".join(random.choices(alphabet, k=n_characters))


class BaseEvent(BaseModel, Generic[T]):
    type: T = None  # type: ignore - will be set by validator below
    event_id: str = Field(default_factory=lambda: random_id("event"))

    @model_validator(mode="after")
    def set_type_from_generic(self) -> "BaseEvent":
        if type(self) == BaseEvent:  # noqa - we want to use type() == and not isinstance
            raise ValueError("Cannot instantiate BaseEvent directly")

        # e.g. Literal["session.update"]
        type_argument = self.__class__.model_fields["type"].annotation

        if get_origin(type_argument) is not Literal:
            # I don't know how to enforce this in the type system directly
            raise ValueError("Type argument is not a Literal")

        self.type = get_args(type_argument)[0]  # e.g. "session.update"

        return self


class ErrorDetails(BaseModel):
    type: str
    code: str | None = None
    message: str
    param: str | None = None
    # ours, not part of the OpenAI API:
    details: object | None = None


class Error(BaseEvent[Literal["error"]]):
    error: ErrorDetails


# =============================================================================
# Tool/Function Calling Types
# =============================================================================


class ToolFunctionParameters(BaseModel):
    """JSON Schema for function parameters."""

    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolFunction(BaseModel):
    """Function definition for a tool."""

    name: str
    description: str = ""
    parameters: ToolFunctionParameters = Field(default_factory=ToolFunctionParameters)


class Tool(BaseModel):
    """Tool definition (currently only function type supported)."""

    type: Literal["function"] = "function"
    # Nested format (OpenAI standard)
    function: ToolFunction | None = None
    # Flat format (also accepted by OpenAI)
    name: str | None = None
    description: str | None = None
    parameters: ToolFunctionParameters | None = None

    def get_name(self) -> str:
        """Get the function name from either format."""
        if self.function:
            return self.function.name
        return self.name or ""

    def get_description(self) -> str:
        """Get the function description from either format."""
        if self.function:
            return self.function.description
        return self.description or ""

    def get_parameters(self) -> ToolFunctionParameters:
        """Get the function parameters from either format."""
        if self.function:
            return self.function.parameters
        return self.parameters or ToolFunctionParameters()

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format for LLM calls."""
        return {
            "type": "function",
            "function": {
                "name": self.get_name(),
                "description": self.get_description(),
                "parameters": self.get_parameters().model_dump(),
            },
        }


class SessionConfig(BaseModel):
    # The "Instructions" object is an Unmute extension
    instructions: Instructions | None = None
    voice: str | None = None
    allow_recording: bool
    # Tool calling support (OpenAI Realtime API compatible)
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None  # "auto", "none", "required", or {"type": "function", "function": {"name": "..."}}


class SessionUpdate(BaseEvent[Literal["session.update"]]):
    session: SessionConfig


class SessionUpdated(BaseEvent[Literal["session.updated"]]):
    session: SessionConfig


class InputAudioBufferAppend(BaseEvent[Literal["input_audio_buffer.append"]]):
    audio: str  # Base64-encoded Opus data


class UnmuteInputAudioBufferAppendAnonymized(
    BaseEvent[Literal["unmute.input_audio_buffer.append_anonymized"]]
):
    """
    For recording, an anonymous version of InputAudioBufferAppend that only says
    how many samples were appended, not the actual audio data.
    """

    number_of_samples: int


class InputAudioBufferSpeechStarted(
    BaseEvent[Literal["input_audio_buffer.speech_started"]]
):
    """Speech started according to the STT.

    Note this is not symmetrical with `InputAudioBufferSpeechStopped` because it's based
    on the STT and not the VAD signal. This is because sometimes the VAD will think
    there is speech but then nothing will end up getting transcribed. If we were using
    the VAD for both events we might get a start event without a stop event.
    For VAD interruptions, see `UnmuteInterruptedByVAD`.
    """


class InputAudioBufferSpeechStopped(
    BaseEvent[Literal["input_audio_buffer.speech_stopped"]]
):
    """A pause was detected by the VAD."""


class Response(BaseModel):
    object: Literal["realtime.response"] = "realtime.response"
    # We currently only use in_progress
    status: Literal["in_progress", "completed", "cancelled", "failed", "incomplete"]
    voice: str
    chat_history: list[dict[str, Any]] = Field(default_factory=list)


class ResponseCreated(BaseEvent[Literal["response.created"]]):
    response: Response


class ResponseTextDelta(BaseEvent[Literal["response.text.delta"]]):
    delta: str


class ResponseTextDone(BaseEvent[Literal["response.text.done"]]):
    text: str


class ResponseAudioDelta(BaseEvent[Literal["response.audio.delta"]]):
    delta: str  # Base64-encoded Opus audio data


class ResponseAudioDone(BaseEvent[Literal["response.audio.done"]]):
    pass


class TranscriptLogprob(BaseModel):
    bytes: bytes
    logprob: float
    token: str


class ConversationItemInputAudioTranscriptionDelta(
    BaseEvent[Literal["conversation.item.input_audio_transcription.delta"]]
):
    delta: str
    start_time: float  # Unmute extension


class UnmuteAdditionalOutputs(BaseEvent[Literal["unmute.additional_outputs"]]):
    args: Any


class UnmuteResponseTextDeltaReady(
    BaseEvent[Literal["unmute.response.text.delta.ready"]]
):
    delta: str


class UnmuteResponseAudioDeltaReady(
    BaseEvent[Literal["unmute.response.audio.delta.ready"]]
):
    number_of_samples: int


class UnmuteInterruptedByVAD(BaseEvent[Literal["unmute.interrupted_by_vad"]]):
    """The VAD interrupted the response generation."""


# =============================================================================
# Function Calling Events (OpenAI Realtime API compatible)
# =============================================================================


class FunctionCallItem(BaseModel):
    """Represents a function call item in a response."""

    id: str = Field(default_factory=lambda: random_id("item"))
    object: Literal["realtime.item"] = "realtime.item"
    type: Literal["function_call"] = "function_call"
    status: Literal["in_progress", "completed"] = "in_progress"
    name: str
    call_id: str = Field(default_factory=lambda: random_id("call"))
    arguments: str = ""


class ResponseOutputItemAdded(BaseEvent[Literal["response.output_item.added"]]):
    """Server emits when a new output item (text or function_call) is being added."""

    response_id: str
    output_index: int
    item: FunctionCallItem | dict[str, Any]


class ResponseFunctionCallArgumentsDelta(
    BaseEvent[Literal["response.function_call_arguments.delta"]]
):
    """Streaming function call arguments as they're generated."""

    response_id: str
    item_id: str
    output_index: int
    call_id: str
    delta: str  # JSON fragment


class ResponseFunctionCallArgumentsDone(
    BaseEvent[Literal["response.function_call_arguments.done"]]
):
    """Function call arguments are complete."""

    response_id: str
    item_id: str
    output_index: int
    call_id: str
    arguments: str  # Complete JSON string


class ResponseOutputItemDone(BaseEvent[Literal["response.output_item.done"]]):
    """Output item (function_call or message) is complete."""

    response_id: str
    output_index: int
    item: FunctionCallItem | dict[str, Any]


class ResponseDone(BaseEvent[Literal["response.done"]]):
    """Response is complete (all output items finished)."""

    response_id: str


# Client events for function call results


class FunctionCallOutputItem(BaseModel):
    """Function call result submitted by client."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str  # JSON string with function result


class ConversationItemCreate(BaseEvent[Literal["conversation.item.create"]]):
    """Client creates a conversation item (e.g., function call result)."""

    item: FunctionCallOutputItem | dict[str, Any]


class ResponseCreate(BaseEvent[Literal["response.create"]]):
    """Client triggers response generation (required after function call result)."""

    pass


# Server events (from OpenAI to client)
ServerEvent = Union[
    Error,
    SessionUpdated,
    ResponseCreated,
    ResponseTextDelta,
    ResponseTextDone,
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseDone,
    # Function calling events
    ResponseOutputItemAdded,
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
    ResponseOutputItemDone,
    # Other events
    ConversationItemInputAudioTranscriptionDelta,
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    # Unmute extensions
    UnmuteAdditionalOutputs,
    UnmuteResponseTextDeltaReady,
    UnmuteResponseAudioDeltaReady,
    UnmuteInterruptedByVAD,
]

# Client events (from client to OpenAI)
ClientEvent = Union[
    SessionUpdate,
    InputAudioBufferAppend,
    # Function calling events
    ConversationItemCreate,
    ResponseCreate,
    # Used internally for recording, we're not expecting the user to send this
    UnmuteInputAudioBufferAppendAnonymized,
]

Event = ClientEvent | ServerEvent
