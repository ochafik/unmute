"""OpenAI Realtime API WebSocket client.

Handles the WebSocket connection to OpenAI's Realtime API endpoint.
Manages authentication, message serialization, and event streaming.
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import websockets
from websockets.asyncio.client import ClientConnection
from pydantic import TypeAdapter, Field
from typing import Annotated

from unmute.openai_realtime.events import (
    ClientEvent,
    ServerEvent,
    SessionConfig,
    SessionUpdateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferClearEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
    ResponseConfig,
    ConversationItemCreateEvent,
    ConversationItem,
)

logger = logging.getLogger(__name__)

# Default OpenAI Realtime API endpoint
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-4o-realtime-preview"

# Type adapter for parsing server events with discriminator
ServerEventAdapter = TypeAdapter(
    Annotated[ServerEvent, Field(discriminator="type")]
)


class OpenAIRealtimeClient:
    """WebSocket client for OpenAI Realtime API.

    Example usage:
        async with OpenAIRealtimeClient(api_key="sk-...") as client:
            await client.update_session(SessionConfig(voice="alloy"))

            # Send audio
            await client.send_audio(pcm16_bytes)

            # Receive events
            async for event in client:
                if event.type == "response.audio.delta":
                    play_audio(event.delta)
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = OPENAI_REALTIME_URL,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._ws: ClientConnection | None = None
        self._receive_queue: asyncio.Queue[ServerEvent] = asyncio.Queue()
        self._receive_task: asyncio.Task | None = None
        self._closed = False

    @property
    def url(self) -> str:
        """Get the full WebSocket URL with model parameter."""
        return f"{self.base_url}?model={self.model}"

    @property
    def headers(self) -> dict[str, str]:
        """Get the authentication headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

    async def connect(self) -> None:
        """Establish WebSocket connection to OpenAI."""
        if self._ws is not None:
            raise RuntimeError("Already connected")

        logger.info(f"Connecting to OpenAI Realtime API: {self.url}")

        self._ws = await websockets.connect(
            self.url,
            additional_headers=self.headers,
            max_size=None,  # No limit on message size
        )

        # Start background task to receive messages
        self._receive_task = asyncio.create_task(
            self._receive_loop(), name="openai_realtime_receive"
        )

        logger.info("Connected to OpenAI Realtime API")

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._closed = True

        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        logger.info("Disconnected from OpenAI Realtime API")

    async def __aenter__(self) -> "OpenAIRealtimeClient":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _receive_loop(self) -> None:
        """Background task to receive messages from OpenAI."""
        assert self._ws is not None

        try:
            async for message in self._ws:
                if self._closed:
                    break

                try:
                    data = json.loads(message)
                    event = ServerEventAdapter.validate_python(data)
                    await self._receive_queue.put(event)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message as JSON: {e}")
                except Exception as e:
                    logger.error(f"Failed to parse server event: {e}, data: {message[:200]}")

        except websockets.ConnectionClosed as e:
            logger.info(f"OpenAI WebSocket closed: {e.code} {e.reason}")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")

    async def send(self, event: ClientEvent) -> None:
        """Send a client event to OpenAI."""
        if self._ws is None:
            raise RuntimeError("Not connected")

        data = event.model_dump(exclude_none=True)
        await self._ws.send(json.dumps(data))
        logger.debug(f"Sent event: {event.type}")

    async def receive(self, timeout: float | None = None) -> ServerEvent | None:
        """Receive a server event from OpenAI.

        Args:
            timeout: Maximum time to wait for an event. None = wait forever.

        Returns:
            ServerEvent or None if timeout reached.
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._receive_queue.get(), timeout)
            else:
                return await self._receive_queue.get()
        except asyncio.TimeoutError:
            return None

    def receive_nowait(self) -> ServerEvent | None:
        """Non-blocking receive of a server event."""
        try:
            return self._receive_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def __aiter__(self) -> AsyncIterator[ServerEvent]:
        """Iterate over server events."""
        while not self._closed:
            event = await self.receive(timeout=0.1)
            if event is not None:
                yield event

    # =========================================================================
    # High-level API methods
    # =========================================================================

    async def update_session(self, config: SessionConfig) -> None:
        """Update session configuration."""
        await self.send(SessionUpdateEvent(session=config))

    async def send_audio(self, audio_base64: str) -> None:
        """Send audio data to the input buffer.

        Args:
            audio_base64: Base64-encoded PCM16 audio at 24kHz mono.
        """
        await self.send(InputAudioBufferAppendEvent(audio=audio_base64))

    async def commit_audio(self) -> None:
        """Commit the input audio buffer to create a user message."""
        await self.send(InputAudioBufferCommitEvent())

    async def clear_audio(self) -> None:
        """Clear the input audio buffer."""
        await self.send(InputAudioBufferClearEvent())

    async def create_response(self, config: ResponseConfig | None = None) -> None:
        """Trigger response generation."""
        await self.send(ResponseCreateEvent(response=config))

    async def cancel_response(self) -> None:
        """Cancel an in-progress response."""
        await self.send(ResponseCancelEvent())

    async def add_item(
        self,
        item: ConversationItem,
        previous_item_id: str | None = None,
    ) -> None:
        """Add an item to the conversation."""
        await self.send(
            ConversationItemCreateEvent(
                item=item,
                previous_item_id=previous_item_id,
            )
        )
