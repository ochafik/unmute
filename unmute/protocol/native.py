"""
Native Unmute protocol adapter.

Implements the original Unmute protocol using Opus-encoded audio at 24kHz.
This adapter maintains 100% backward compatibility with existing clients.
"""

import asyncio
import base64
from typing import Optional, Tuple, Union

import numpy as np
import sphn
from fastrtc import AdditionalOutputs, CloseStream, audio_to_float32
from pydantic import TypeAdapter, ValidationError

import unmute.openai_realtime_api_events as ora
from unmute.protocol.base import ProtocolAdapter


class NativeProtocolAdapter(ProtocolAdapter):
    """
    Native Unmute protocol adapter.

    Uses Opus codec for efficient audio transmission (94% bandwidth savings vs PCM16).
    Message format is OpenAI Realtime API-inspired with Unmute-specific extensions.
    """

    def __init__(self, sample_rate: int = 24000):
        super().__init__(sample_rate)
        self.opus_reader: Optional[sphn.OpusStreamReader] = None
        self.opus_writer: Optional[sphn.OpusStreamWriter] = None
        self.client_event_adapter = TypeAdapter(ora.ClientEvent)

    async def setup(self):
        """Set up Opus codecs for audio encoding/decoding."""
        # Create Opus reader for decoding client audio (Opus → Float32)
        self.opus_reader = await asyncio.to_thread(
            sphn.OpusStreamReader,
            self.sample_rate,
        )

        # Create Opus writer for encoding server audio (Float32 → Opus)
        self.opus_writer = await asyncio.to_thread(
            sphn.OpusStreamWriter,
            self.sample_rate,
        )

    async def teardown(self):
        """Clean up Opus codecs."""
        # Opus codecs don't require explicit cleanup
        self.opus_reader = None
        self.opus_writer = None

    async def decode_audio(self, audio_data: str) -> Optional[np.ndarray]:
        """
        Decode Opus audio to PCM float32.

        Args:
            audio_data: Base64-encoded Opus bytes

        Returns:
            PCM float32 audio array, or None if no audio produced
        """
        if self.opus_reader is None:
            raise RuntimeError("NativeProtocolAdapter not set up - call setup() first")

        # Decode base64 to Opus bytes
        opus_bytes = base64.b64decode(audio_data)

        # Decode Opus to PCM float32
        pcm = await asyncio.to_thread(self.opus_reader.append_bytes, opus_bytes)

        if pcm.size > 0:
            return pcm
        return None

    async def encode_audio(self, audio: np.ndarray) -> Optional[str]:
        """
        Encode PCM float32 audio to Opus.

        Args:
            audio: PCM float32 audio array

        Returns:
            Base64-encoded Opus bytes, or None if no output
        """
        if self.opus_writer is None:
            raise RuntimeError("NativeProtocolAdapter not set up - call setup() first")

        # Ensure float32 format
        audio = audio_to_float32(audio)

        # Encode PCM float32 to Opus
        opus_bytes = await asyncio.to_thread(self.opus_writer.append_pcm, audio)

        if opus_bytes:
            # Encode to base64
            return base64.b64encode(opus_bytes).decode("utf-8")
        return None

    def translate_client_message(self, message_json: str) -> Union[ora.ClientEvent, None]:
        """
        Parse and validate client message in native Unmute format.

        Args:
            message_json: Raw JSON string from client

        Returns:
            Parsed ClientEvent object

        Raises:
            ValidationError: If message is invalid
        """
        # Native protocol uses the same message format as internal representation
        # No translation needed - just validate
        return self.client_event_adapter.validate_json(message_json)

    def translate_server_event(
        self,
        event: Union[ora.ServerEvent, Tuple[int, np.ndarray], AdditionalOutputs, CloseStream]
    ) -> Optional[str]:
        """
        Translate server event to native Unmute format.

        Args:
            event: Internal event object

        Returns:
            JSON string to send to client, or None if nothing to send
        """
        # Handle different event types
        if isinstance(event, AdditionalOutputs):
            # Debug outputs - wrap in UnmuteAdditionalOutputs event
            return ora.UnmuteAdditionalOutputs(args=event.args[0] if event.args else {}).model_dump_json()

        elif isinstance(event, CloseStream):
            # Connection close - handled by WebSocket layer, nothing to send
            return None

        elif isinstance(event, ora.ServerEvent):
            # Native protocol uses the same message format as internal representation
            # No translation needed - just serialize
            return event.model_dump_json()

        elif isinstance(event, tuple):
            # Audio tuple - should be handled by handle_audio_output, not here
            raise ValueError("Audio tuples should be handled by handle_audio_output()")

        else:
            raise TypeError(f"Unknown event type: {type(event)}")

    @property
    def protocol_name(self) -> str:
        return "native"

    @property
    def websocket_subprotocol(self) -> str:
        return "realtime"

    def get_session_config(self) -> dict:
        """Get native protocol session configuration."""
        return {
            "audio_codec": "opus",
            "sample_rate": self.sample_rate,
            "bandwidth_multiplier": 0.06,  # Opus uses ~6% of PCM16 bandwidth
        }
