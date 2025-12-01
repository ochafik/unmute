"""OpenAI Realtime API handler for Unmute.

This handler bridges the browser WebSocket connection to OpenAI's Realtime API.
It supports two audio modes:
1. PCM16 mode (default): Browser sends/receives PCM16 directly - optimal latency
2. Opus mode: Browser sends/receives Opus, handler converts - backward compatible

Architecture:
    Browser <--WebSocket--> UnmuteBackend <--WebSocket--> OpenAI Realtime API

The handler translates between Unmute's protocol and OpenAI's protocol,
allowing the frontend to work with either local STT/TTS or OpenAI's native
voice capabilities.
"""

import asyncio
import base64
import logging
from typing import Any

import numpy as np

from unmute.openai_realtime.client import OpenAIRealtimeClient
from unmute.openai_realtime.events import (
    SessionConfig,
    InputAudioTranscription,
    ServerVADConfig,
    Voice,
    # Server events
    SessionCreatedEvent,
    SessionUpdatedEvent,
    ErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    RateLimitsUpdatedEvent,
)
import unmute.openai_realtime_api_events as ora

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2  # PCM16

# Default OpenAI voice mapping (Unmute voice name -> OpenAI voice)
DEFAULT_VOICE: Voice = "alloy"


class OpenAIRealtimeHandler:
    """Handler that bridges browser to OpenAI Realtime API.

    This handler:
    1. Receives audio from browser (PCM16 or Opus)
    2. Forwards to OpenAI Realtime API
    3. Receives audio/text responses from OpenAI
    4. Forwards to browser in appropriate format

    Audio Format Modes:
    - "pcm16": Browser sends/receives base64 PCM16 (optimal, no conversion)
    - "opus": Browser sends/receives Opus (requires conversion, backward compat)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-realtime-preview",
        audio_format: str = "pcm16",  # "pcm16" or "opus"
    ):
        self.api_key = api_key
        self.model = model
        self.audio_format = audio_format

        self._client: OpenAIRealtimeClient | None = None
        self._output_queue: asyncio.Queue[ora.ServerEvent | tuple[int, np.ndarray]] = (
            asyncio.Queue()
        )

        # Session state
        self._session_config: SessionConfig | None = None
        self._instructions: str | None = None
        self._voice: Voice = DEFAULT_VOICE

        # For Opus mode conversion (lazy loaded)
        self._opus_reader: Any = None
        self._opus_writer: Any = None

    async def start_up(self) -> None:
        """Initialize and connect to OpenAI Realtime API."""
        self._client = OpenAIRealtimeClient(
            api_key=self.api_key,
            model=self.model,
        )
        await self._client.connect()

        # Start listening for OpenAI events
        asyncio.create_task(self._openai_event_loop(), name="openai_event_loop")

        logger.info(f"OpenAI Realtime handler started (audio_format={self.audio_format})")

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "OpenAIRealtimeHandler":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.cleanup()

    async def update_session(self, session: ora.SessionConfig) -> None:
        """Update session configuration from Unmute protocol.

        Translates Unmute's session config to OpenAI's format.
        """
        if self._client is None:
            raise RuntimeError("Handler not started")

        # Extract instructions
        if session.instructions:
            # Unmute uses an Instructions object, OpenAI expects a string
            if hasattr(session.instructions, "model_dump"):
                # Convert Instructions object to a prompt string
                instr = session.instructions
                parts = []
                if instr.ai_personality:
                    parts.append(f"Personality: {instr.ai_personality}")
                if instr.conversation_topic:
                    parts.append(f"Topic: {instr.conversation_topic}")
                if instr.user_description:
                    parts.append(f"User: {instr.user_description}")
                self._instructions = "\n".join(parts) if parts else None
            else:
                self._instructions = str(session.instructions)

        # Map Unmute voice to OpenAI voice
        if session.voice:
            # Check if it's a valid OpenAI voice, otherwise use default
            openai_voices = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
            if session.voice.lower() in openai_voices:
                self._voice = session.voice.lower()  # type: ignore
            else:
                # Use default voice for custom Unmute voices
                logger.info(f"Voice '{session.voice}' not available in OpenAI, using '{DEFAULT_VOICE}'")
                self._voice = DEFAULT_VOICE

        # Build OpenAI session config
        openai_config = SessionConfig(
            modalities=["text", "audio"],
            instructions=self._instructions,
            voice=self._voice,
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            input_audio_transcription=InputAudioTranscription(model="whisper-1"),
            turn_detection=ServerVADConfig(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=500,
                create_response=True,
            ),
            temperature=0.8,
        )

        await self._client.update_session(openai_config)
        self._session_config = openai_config

    async def receive_audio(self, audio_data: str) -> None:
        """Receive audio from browser and forward to OpenAI.

        Args:
            audio_data: Base64-encoded audio (PCM16 or Opus depending on mode)
        """
        if self._client is None:
            raise RuntimeError("Handler not started")

        if self.audio_format == "pcm16":
            # Direct passthrough - no conversion needed
            await self._client.send_audio(audio_data)
        else:
            # Opus mode - convert to PCM16
            pcm16_base64 = await self._opus_to_pcm16(audio_data)
            await self._client.send_audio(pcm16_base64)

    async def emit(self) -> ora.ServerEvent | tuple[int, np.ndarray] | None:
        """Get next event to send to browser.

        Returns events in Unmute's protocol format.
        """
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            # Small sleep to prevent busy loop
            await asyncio.sleep(0.001)
            return None

    async def _openai_event_loop(self) -> None:
        """Background task to process events from OpenAI."""
        if self._client is None:
            return

        try:
            async for event in self._client:
                await self._handle_openai_event(event)
        except asyncio.CancelledError:
            logger.info("OpenAI event loop cancelled")
        except Exception as e:
            logger.error(f"Error in OpenAI event loop: {e}")
            # Send error to client
            await self._output_queue.put(
                ora.Error(
                    error=ora.ErrorDetails(
                        type="openai_error",
                        message=str(e),
                    )
                )
            )

    async def _handle_openai_event(self, event: Any) -> None:
        """Handle an event from OpenAI and translate to Unmute protocol."""

        if isinstance(event, SessionCreatedEvent):
            logger.info(f"OpenAI session created: {event.session.id}")
            # Session created is handled internally, send session.updated to client
            if self._session_config:
                await self._output_queue.put(
                    ora.SessionUpdated(
                        session=ora.SessionConfig(
                            instructions=None,  # Don't echo back
                            voice=self._voice,
                            allow_recording=False,  # OpenAI mode doesn't support recording
                        )
                    )
                )

        elif isinstance(event, SessionUpdatedEvent):
            logger.debug("OpenAI session updated")

        elif isinstance(event, ErrorEvent):
            logger.error(f"OpenAI error: {event.error.message}")
            await self._output_queue.put(
                ora.Error(
                    error=ora.ErrorDetails(
                        type=event.error.type,
                        code=event.error.code,
                        message=event.error.message,
                    )
                )
            )

        elif isinstance(event, ResponseAudioDeltaEvent):
            # Audio from OpenAI - forward to browser
            if self.audio_format == "pcm16":
                # Direct passthrough
                await self._output_queue.put(
                    ora.ResponseAudioDelta(delta=event.delta)
                )
            else:
                # Convert PCM16 to Opus
                opus_base64 = await self._pcm16_to_opus(event.delta)
                await self._output_queue.put(
                    ora.ResponseAudioDelta(delta=opus_base64)
                )

        elif isinstance(event, ResponseAudioDoneEvent):
            await self._output_queue.put(ora.ResponseAudioDone())

        elif isinstance(event, ResponseTextDeltaEvent):
            await self._output_queue.put(
                ora.ResponseTextDelta(delta=event.delta)
            )

        elif isinstance(event, ResponseTextDoneEvent):
            await self._output_queue.put(
                ora.ResponseTextDone(text=event.text)
            )

        elif isinstance(event, ResponseAudioTranscriptDeltaEvent):
            # This is the assistant's speech transcript (what the AI is saying)
            # We can use ResponseTextDelta for this as well
            pass  # Already handled by ResponseTextDelta

        elif isinstance(event, ResponseAudioTranscriptDoneEvent):
            pass  # Already handled by ResponseTextDone

        elif isinstance(event, ConversationItemInputAudioTranscriptionCompletedEvent):
            # User's speech transcription
            await self._output_queue.put(
                ora.ConversationItemInputAudioTranscriptionDelta(
                    delta=event.transcript,
                    start_time=0.0,  # OpenAI doesn't provide timing
                )
            )

        elif isinstance(event, InputAudioBufferSpeechStartedEvent):
            await self._output_queue.put(ora.InputAudioBufferSpeechStarted())

        elif isinstance(event, InputAudioBufferSpeechStoppedEvent):
            await self._output_queue.put(ora.InputAudioBufferSpeechStopped())

        elif isinstance(event, ResponseCreatedEvent):
            await self._output_queue.put(
                ora.ResponseCreated(
                    response=ora.Response(
                        status="in_progress",
                        voice=self._voice,
                        chat_history=[],
                    )
                )
            )

        elif isinstance(event, ResponseDoneEvent):
            logger.debug(f"Response done: {event.response.status}")

        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            # Function calling - log for now, can be extended
            logger.info(f"Function call: {event.name}({event.arguments})")

        elif isinstance(event, RateLimitsUpdatedEvent):
            logger.debug(f"Rate limits: {event.rate_limits}")

        else:
            # Log unhandled events for debugging
            event_type = getattr(event, "type", type(event).__name__)
            logger.debug(f"Unhandled OpenAI event: {event_type}")

    # =========================================================================
    # Audio Conversion (for Opus mode)
    # =========================================================================

    async def _opus_to_pcm16(self, opus_base64: str) -> str:
        """Convert Opus audio to PCM16 base64."""
        import sphn

        if self._opus_reader is None:
            self._opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

        opus_bytes = base64.b64decode(opus_base64)
        pcm_float = await asyncio.to_thread(self._opus_reader.append_bytes, opus_bytes)

        if pcm_float.size == 0:
            return ""

        # Convert float32 [-1, 1] to int16
        pcm_int16 = (pcm_float * 32767).astype(np.int16)
        return base64.b64encode(pcm_int16.tobytes()).decode("utf-8")

    async def _pcm16_to_opus(self, pcm16_base64: str) -> str:
        """Convert PCM16 base64 to Opus."""
        import sphn

        if self._opus_writer is None:
            self._opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)

        pcm_bytes = base64.b64decode(pcm16_base64)
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Convert int16 to float32 [-1, 1]
        pcm_float = pcm_int16.astype(np.float32) / 32767.0

        opus_bytes = await asyncio.to_thread(self._opus_writer.append_pcm, pcm_float)

        if not opus_bytes:
            return ""

        return base64.b64encode(opus_bytes).decode("utf-8")
