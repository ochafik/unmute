"""OpenAI Realtime API event types.

Complete type definitions for the OpenAI Realtime API protocol.
Reference: https://platform.openai.com/docs/api-reference/realtime

Audio Format: PCM16 at 24kHz, mono, little-endian, base64 encoded
"""

from typing import Any, Literal, Union
from pydantic import BaseModel, Field
import random


def random_id(prefix: str = "evt") -> str:
    """Generate a random event ID like OpenAI does."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return f"{prefix}_" + "".join(random.choices(alphabet, k=21))


# =============================================================================
# Audio Formats
# =============================================================================

AudioFormat = Literal["pcm16", "g711_ulaw", "g711_alaw"]

# =============================================================================
# Voice Options
# =============================================================================

Voice = Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]

# =============================================================================
# Modalities
# =============================================================================

Modality = Literal["text", "audio"]

# =============================================================================
# Turn Detection Configuration
# =============================================================================


class ServerVADConfig(BaseModel):
    """Server-side Voice Activity Detection configuration."""

    type: Literal["server_vad"] = "server_vad"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    prefix_padding_ms: int = Field(default=300, ge=0)
    silence_duration_ms: int = Field(default=500, ge=0)
    create_response: bool = True


class SemanticVADConfig(BaseModel):
    """Semantic VAD - uses model to detect end of utterance."""

    type: Literal["semantic_vad"] = "semantic_vad"
    eagerness: Literal["low", "medium", "high", "auto"] = "auto"
    create_response: bool = True


class NoTurnDetection(BaseModel):
    """Disable automatic turn detection."""

    type: Literal["none"] = "none"


TurnDetectionConfig = Union[ServerVADConfig, SemanticVADConfig, NoTurnDetection]

# =============================================================================
# Tool / Function Definitions
# =============================================================================


class FunctionParameters(BaseModel):
    """JSON Schema for function parameters."""

    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class FunctionTool(BaseModel):
    """Function tool definition."""

    type: Literal["function"] = "function"
    name: str
    description: str
    parameters: FunctionParameters


Tool = FunctionTool  # May expand to other tool types in future

# =============================================================================
# Input Audio Transcription Config
# =============================================================================


class InputAudioTranscription(BaseModel):
    """Configuration for transcribing user audio input."""

    model: str = "whisper-1"


# =============================================================================
# Session Configuration
# =============================================================================


class SessionConfig(BaseModel):
    """Session configuration sent via session.update."""

    modalities: list[Modality] = Field(default_factory=lambda: ["text", "audio"])
    instructions: str | None = None
    voice: Voice = "alloy"
    input_audio_format: AudioFormat = "pcm16"
    output_audio_format: AudioFormat = "pcm16"
    input_audio_transcription: InputAudioTranscription | None = None
    turn_detection: TurnDetectionConfig | None = Field(
        default_factory=lambda: ServerVADConfig()
    )
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    temperature: float = Field(default=0.8, ge=0.6, le=1.2)
    max_response_output_tokens: int | Literal["inf"] = "inf"


class SessionResource(BaseModel):
    """Session resource returned by server."""

    id: str
    object: Literal["realtime.session"] = "realtime.session"
    model: str
    expires_at: int
    modalities: list[Modality]
    instructions: str | None = None
    voice: Voice
    input_audio_format: AudioFormat
    output_audio_format: AudioFormat
    input_audio_transcription: InputAudioTranscription | None = None
    turn_detection: TurnDetectionConfig | None = None
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: str = "auto"
    temperature: float
    max_response_output_tokens: int | str


# =============================================================================
# Conversation Items
# =============================================================================

ItemType = Literal["message", "function_call", "function_call_output"]
ItemRole = Literal["user", "assistant", "system"]
ItemStatus = Literal["completed", "incomplete", "in_progress"]
ContentType = Literal["input_text", "input_audio", "item_reference", "text", "audio"]


class ContentPart(BaseModel):
    """Content part within a conversation item."""

    type: ContentType
    text: str | None = None
    audio: str | None = None  # Base64 encoded PCM16
    transcript: str | None = None


class ConversationItem(BaseModel):
    """A conversation item (message, function call, etc.)."""

    id: str = Field(default_factory=lambda: random_id("item"))
    object: Literal["realtime.item"] = "realtime.item"
    type: ItemType = "message"
    status: ItemStatus = "completed"
    role: ItemRole | None = None
    content: list[ContentPart] = Field(default_factory=list)
    call_id: str | None = None  # For function calls
    name: str | None = None  # Function name
    arguments: str | None = None  # Function arguments JSON
    output: str | None = None  # Function output


# =============================================================================
# Response Configuration
# =============================================================================


class ResponseConfig(BaseModel):
    """Configuration for response.create event."""

    modalities: list[Modality] | None = None
    instructions: str | None = None
    voice: Voice | None = None
    output_audio_format: AudioFormat | None = None
    tools: list[Tool] | None = None
    tool_choice: Literal["auto", "none", "required"] | str | None = None
    temperature: float | None = None
    max_output_tokens: int | Literal["inf"] | None = None
    conversation: Literal["auto", "none"] = "auto"
    input: list[ConversationItem] | None = None  # For out-of-band responses


class ResponseResource(BaseModel):
    """Response resource from server."""

    id: str
    object: Literal["realtime.response"] = "realtime.response"
    status: Literal["in_progress", "completed", "cancelled", "incomplete", "failed"]
    status_details: dict[str, Any] | None = None
    output: list[ConversationItem] = Field(default_factory=list)
    usage: dict[str, Any] | None = None


# =============================================================================
# Rate Limits
# =============================================================================


class RateLimit(BaseModel):
    """Rate limit information."""

    name: str
    limit: int
    remaining: int
    reset_seconds: float


# =============================================================================
# CLIENT EVENTS (Browser/App → OpenAI)
# =============================================================================


class ClientEventBase(BaseModel):
    """Base class for client events."""

    event_id: str = Field(default_factory=lambda: random_id("event"))


# --- Session Events ---


class SessionUpdateEvent(ClientEventBase):
    """Update session configuration."""

    type: Literal["session.update"] = "session.update"
    session: SessionConfig


# --- Input Audio Buffer Events ---


class InputAudioBufferAppendEvent(ClientEventBase):
    """Append audio to the input buffer."""

    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str  # Base64-encoded PCM16 audio


class InputAudioBufferCommitEvent(ClientEventBase):
    """Commit the input audio buffer."""

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class InputAudioBufferClearEvent(ClientEventBase):
    """Clear the input audio buffer."""

    type: Literal["input_audio_buffer.clear"] = "input_audio_buffer.clear"


# --- Conversation Item Events ---


class ConversationItemCreateEvent(ClientEventBase):
    """Create a conversation item."""

    type: Literal["conversation.item.create"] = "conversation.item.create"
    previous_item_id: str | None = None
    item: ConversationItem


class ConversationItemTruncateEvent(ClientEventBase):
    """Truncate a conversation item's audio."""

    type: Literal["conversation.item.truncate"] = "conversation.item.truncate"
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDeleteEvent(ClientEventBase):
    """Delete a conversation item."""

    type: Literal["conversation.item.delete"] = "conversation.item.delete"
    item_id: str


# --- Response Events ---


class ResponseCreateEvent(ClientEventBase):
    """Create a response."""

    type: Literal["response.create"] = "response.create"
    response: ResponseConfig | None = None


class ResponseCancelEvent(ClientEventBase):
    """Cancel an in-progress response."""

    type: Literal["response.cancel"] = "response.cancel"


# Union of all client events
ClientEvent = Union[
    SessionUpdateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferClearEvent,
    ConversationItemCreateEvent,
    ConversationItemTruncateEvent,
    ConversationItemDeleteEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
]

# =============================================================================
# SERVER EVENTS (OpenAI → Browser/App)
# =============================================================================


class ServerEventBase(BaseModel):
    """Base class for server events."""

    event_id: str


# --- Error Event ---


class ErrorDetail(BaseModel):
    """Error detail information."""

    type: str
    code: str | None = None
    message: str
    param: str | None = None
    event_id: str | None = None


class ErrorEvent(ServerEventBase):
    """Error event from server."""

    type: Literal["error"] = "error"
    error: ErrorDetail


# --- Session Events ---


class SessionCreatedEvent(ServerEventBase):
    """Session created event."""

    type: Literal["session.created"] = "session.created"
    session: SessionResource


class SessionUpdatedEvent(ServerEventBase):
    """Session updated event."""

    type: Literal["session.updated"] = "session.updated"
    session: SessionResource


# --- Conversation Events ---


class ConversationCreatedEvent(ServerEventBase):
    """Conversation created event."""

    type: Literal["conversation.created"] = "conversation.created"
    conversation: dict[str, Any]


class ConversationItemCreatedEvent(ServerEventBase):
    """Conversation item created event."""

    type: Literal["conversation.item.created"] = "conversation.item.created"
    previous_item_id: str | None = None
    item: ConversationItem


class ConversationItemDeletedEvent(ServerEventBase):
    """Conversation item deleted event."""

    type: Literal["conversation.item.deleted"] = "conversation.item.deleted"
    item_id: str


class ConversationItemTruncatedEvent(ServerEventBase):
    """Conversation item truncated event."""

    type: Literal["conversation.item.truncated"] = "conversation.item.truncated"
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemInputAudioTranscriptionCompletedEvent(ServerEventBase):
    """Input audio transcription completed."""

    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    item_id: str
    content_index: int
    transcript: str


class ConversationItemInputAudioTranscriptionFailedEvent(ServerEventBase):
    """Input audio transcription failed."""

    type: Literal["conversation.item.input_audio_transcription.failed"] = (
        "conversation.item.input_audio_transcription.failed"
    )
    item_id: str
    content_index: int
    error: ErrorDetail


# --- Input Audio Buffer Events ---


class InputAudioBufferCommittedEvent(ServerEventBase):
    """Input audio buffer committed."""

    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    previous_item_id: str | None = None
    item_id: str


class InputAudioBufferClearedEvent(ServerEventBase):
    """Input audio buffer cleared."""

    type: Literal["input_audio_buffer.cleared"] = "input_audio_buffer.cleared"


class InputAudioBufferSpeechStartedEvent(ServerEventBase):
    """Speech started in input audio buffer."""

    type: Literal["input_audio_buffer.speech_started"] = (
        "input_audio_buffer.speech_started"
    )
    audio_start_ms: int
    item_id: str


class InputAudioBufferSpeechStoppedEvent(ServerEventBase):
    """Speech stopped in input audio buffer."""

    type: Literal["input_audio_buffer.speech_stopped"] = (
        "input_audio_buffer.speech_stopped"
    )
    audio_end_ms: int
    item_id: str | None = None


# --- Response Events ---


class ResponseCreatedEvent(ServerEventBase):
    """Response created event."""

    type: Literal["response.created"] = "response.created"
    response: ResponseResource


class ResponseDoneEvent(ServerEventBase):
    """Response done event."""

    type: Literal["response.done"] = "response.done"
    response: ResponseResource


class ResponseOutputItemAddedEvent(ServerEventBase):
    """Response output item added."""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    response_id: str
    output_index: int
    item: ConversationItem


class ResponseOutputItemDoneEvent(ServerEventBase):
    """Response output item done."""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    response_id: str
    output_index: int
    item: ConversationItem


class ResponseContentPartAddedEvent(ServerEventBase):
    """Response content part added."""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: ContentPart


class ResponseContentPartDoneEvent(ServerEventBase):
    """Response content part done."""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: ContentPart


# --- Streaming Delta Events ---


class ResponseTextDeltaEvent(ServerEventBase):
    """Response text delta (streaming)."""

    type: Literal["response.text.delta"] = "response.text.delta"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseTextDoneEvent(ServerEventBase):
    """Response text done."""

    type: Literal["response.text.done"] = "response.text.done"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseAudioTranscriptDeltaEvent(ServerEventBase):
    """Response audio transcript delta (streaming)."""

    type: Literal["response.audio_transcript.delta"] = "response.audio_transcript.delta"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseAudioTranscriptDoneEvent(ServerEventBase):
    """Response audio transcript done."""

    type: Literal["response.audio_transcript.done"] = "response.audio_transcript.done"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    transcript: str


class ResponseAudioDeltaEvent(ServerEventBase):
    """Response audio delta (streaming)."""

    type: Literal["response.audio.delta"] = "response.audio.delta"
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str  # Base64-encoded PCM16 audio


class ResponseAudioDoneEvent(ServerEventBase):
    """Response audio done."""

    type: Literal["response.audio.done"] = "response.audio.done"
    response_id: str
    item_id: str
    output_index: int
    content_index: int


# --- Function Call Events ---


class ResponseFunctionCallArgumentsDeltaEvent(ServerEventBase):
    """Function call arguments delta (streaming)."""

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(ServerEventBase):
    """Function call arguments done."""

    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    name: str
    arguments: str


# --- Rate Limits ---


class RateLimitsUpdatedEvent(ServerEventBase):
    """Rate limits updated event."""

    type: Literal["rate_limits.updated"] = "rate_limits.updated"
    rate_limits: list[RateLimit]


# Union of all server events
ServerEvent = Union[
    ErrorEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    ConversationCreatedEvent,
    ConversationItemCreatedEvent,
    ConversationItemDeletedEvent,
    ConversationItemTruncatedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    RateLimitsUpdatedEvent,
]

# All events
Event = Union[ClientEvent, ServerEvent]
