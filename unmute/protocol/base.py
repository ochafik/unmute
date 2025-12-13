"""
Base protocol adapter interface.

Defines the abstract interface that all protocol adapters must implement
to translate between external protocol formats and Unmute's internal representation.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import numpy as np
from fastrtc import AdditionalOutputs, CloseStream

import unmute.openai_realtime_api_events as ora


class ProtocolAdapter(ABC):
    """
    Abstract base class for protocol adapters.

    Protocol adapters translate between external protocol formats (e.g., OpenAI, RTVI)
    and Unmute's internal representation. Each adapter handles:
    - Audio encoding/decoding
    - Message format translation
    - Protocol-specific handshakes and lifecycle management
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the protocol adapter.

        Args:
            sample_rate: Audio sample rate in Hz (default: 24000)
        """
        self.sample_rate = sample_rate

    @abstractmethod
    async def setup(self):
        """
        Set up the protocol adapter.

        Called once per connection before any messages are processed.
        Used for initializing audio codecs, establishing connections, etc.
        """
        pass

    @abstractmethod
    async def teardown(self):
        """
        Clean up the protocol adapter.

        Called when the connection is closing.
        Used for cleaning up resources, closing codecs, etc.
        """
        pass

    @abstractmethod
    async def decode_audio(self, audio_data: str) -> Optional[np.ndarray]:
        """
        Decode audio from the external protocol format to internal format.

        Args:
            audio_data: Base64-encoded audio in protocol-specific format

        Returns:
            PCM float32 audio array (shape: [samples]) or None if no audio produced
        """
        pass

    @abstractmethod
    async def encode_audio(self, audio: np.ndarray) -> Optional[str]:
        """
        Encode audio from internal format to the external protocol format.

        Args:
            audio: PCM float32 audio array (shape: [samples])

        Returns:
            Base64-encoded audio in protocol-specific format, or None if no output
        """
        pass

    @abstractmethod
    def translate_client_message(self, message_json: str) -> Union[ora.ClientEvent, None]:
        """
        Translate a client message from external protocol to internal format.

        Args:
            message_json: Raw JSON string from client

        Returns:
            Internal ClientEvent object, or None if the message should be ignored

        Raises:
            ValidationError: If the message is invalid
        """
        pass

    @abstractmethod
    def translate_server_event(
        self,
        event: Union[ora.ServerEvent, Tuple[int, np.ndarray], AdditionalOutputs, CloseStream]
    ) -> Optional[str]:
        """
        Translate a server event from internal format to external protocol.

        Args:
            event: Internal event (ServerEvent, audio tuple, or control message)

        Returns:
            JSON string in protocol-specific format, or None if nothing to send
        """
        pass

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Return the name of this protocol (e.g., 'native', 'openai', 'rtvi')."""
        pass

    @property
    @abstractmethod
    def websocket_subprotocol(self) -> str:
        """Return the WebSocket subprotocol string for this protocol."""
        pass

    def get_session_config(self) -> dict[str, Any]:
        """
        Get protocol-specific session configuration.

        Returns:
            Dictionary of configuration values specific to this protocol
        """
        return {}

    async def handle_audio_message(self, message: ora.InputAudioBufferAppend) -> Optional[Tuple[int, np.ndarray]]:
        """
        Handle an audio message and decode it for the handler.

        Args:
            message: InputAudioBufferAppend message with audio data

        Returns:
            Tuple of (sample_rate, audio_array) for UnmuteHandler, or None
        """
        audio = await self.decode_audio(message.audio)
        if audio is not None and audio.size > 0:
            return (self.sample_rate, audio[np.newaxis, :])
        return None

    async def handle_audio_output(self, audio_tuple: Tuple[int, np.ndarray]) -> Optional[ora.ResponseAudioDelta]:
        """
        Handle audio output from the handler and encode it for the client.

        Args:
            audio_tuple: Tuple of (sample_rate, audio_array) from UnmuteHandler

        Returns:
            ResponseAudioDelta message with encoded audio, or None
        """
        _sr, audio = audio_tuple
        # Ensure float32 format (audio_array might be 2D with batch dimension)
        if audio.ndim == 2:
            audio = audio[0]  # Remove batch dimension

        encoded = await self.encode_audio(audio)
        if encoded is not None:
            return ora.ResponseAudioDelta(delta=encoded)
        return None
