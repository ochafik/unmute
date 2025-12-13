"""
OpenAI Realtime API protocol adapter.

Implements OpenAI Realtime API compatibility using PCM16-encoded audio at 24kHz.
This adapter allows OpenAI Realtime API clients (official SDKs, community tools)
to connect to Unmute as if it were OpenAI's server.

Key differences from native protocol:
- Audio: PCM16 (16-bit signed integers) instead of Opus
- Bandwidth: ~16x higher than Opus (384 kbps vs 24 kbps)
- Voice mapping: OpenAI voice names → Kyutai voices
- Session: Expects session.create or model parameter in URL
"""

import json
from typing import Optional, Tuple, Union

import numpy as np
from fastrtc import AdditionalOutputs, CloseStream
from pydantic import TypeAdapter, ValidationError

import unmute.openai_realtime_api_events as ora
from unmute.protocol.audio_transcoding import decode_pcm16_base64, encode_pcm16_base64
from unmute.protocol.base import ProtocolAdapter

# OpenAI voice names that should be mapped to Kyutai "default" voice
# These are the standard OpenAI voices. If the client sends one of these,
# we map it to "default" since Kyutai doesn't have equivalents.
# However, if the client sends a real Kyutai voice name (fetched from /v1/voices),
# we pass it through unchanged.
OPENAI_VOICE_NAMES = {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"}

# For backwards compatibility, map OpenAI voice names to Kyutai "default"
OPENAI_VOICE_MAPPING = {
    "alloy": "default",  # Neutral, balanced
    "ash": "default",  # Clear, expressive (OpenAI preview)
    "ballad": "default",  # Deep, calm (OpenAI preview)
    "coral": "default",  # Warm, friendly (OpenAI preview)
    "echo": "default",  # Resonant, clear
    "sage": "default",  # Soft, articulate
    "shimmer": "default",  # Energetic, bright
    "verse": "default",  # Conversational, engaging (OpenAI preview)
}


class OpenAIProtocolAdapter(ProtocolAdapter):
    """
    OpenAI Realtime API protocol adapter.

    Translates between OpenAI's PCM16 format and Unmute's internal Float32 format.
    Handles OpenAI-specific message patterns and voice mapping.
    """

    def __init__(self, sample_rate: int = 24000, model: str = "gpt-4o-realtime-preview"):
        super().__init__(sample_rate)
        self.model = model
        self.client_event_adapter = TypeAdapter(ora.ClientEvent)
        self._session_configured = False

    async def setup(self):
        """Set up OpenAI protocol adapter."""
        # OpenAI protocol doesn't require codec setup (uses raw PCM16)
        pass

    async def teardown(self):
        """Clean up OpenAI protocol adapter."""
        # No cleanup needed for PCM16
        pass

    async def decode_audio(self, audio_data: str) -> Optional[np.ndarray]:
        """
        Decode PCM16 audio from OpenAI client to Float32.

        Args:
            audio_data: Base64-encoded PCM16 bytes

        Returns:
            PCM Float32 audio array, or None if empty
        """
        try:
            # Decode base64 PCM16 to Float32
            pcm_float32 = decode_pcm16_base64(audio_data)

            if pcm_float32.size > 0:
                return pcm_float32
            return None

        except Exception as e:
            raise ValueError(f"Failed to decode PCM16 audio: {e}")

    async def encode_audio(self, audio: np.ndarray) -> Optional[str]:
        """
        Encode Float32 audio to PCM16 for OpenAI client.

        Args:
            audio: PCM Float32 audio array

        Returns:
            Base64-encoded PCM16 bytes, or None if empty
        """
        try:
            if audio.size == 0:
                return None

            # Encode Float32 to base64 PCM16
            return encode_pcm16_base64(audio)

        except Exception as e:
            raise ValueError(f"Failed to encode PCM16 audio: {e}")

    def _map_openai_voice_to_kyutai(self, voice: str) -> str:
        """
        Map OpenAI voice name to Kyutai voice name.

        If the voice is a standard OpenAI voice name (alloy, echo, etc.),
        map it to "default". Otherwise, assume it's a real Kyutai voice
        name (fetched from /v1/voices) and pass it through unchanged.

        Args:
            voice: Voice name (OpenAI name like "alloy" or Kyutai name)

        Returns:
            Kyutai voice name
        """
        # Only map standard OpenAI voice names to "default"
        # Pass through all other voice names (assumed to be real Kyutai voices)
        if voice in OPENAI_VOICE_NAMES:
            return OPENAI_VOICE_MAPPING.get(voice, "default")
        return voice

    def _map_kyutai_voice_to_openai(self, voice: str) -> str:
        """
        Map Kyutai voice name back for OpenAI client responses.

        If the voice is a standard OpenAI voice name, return it.
        If it's "default", return "alloy".
        Otherwise, it's a real Kyutai voice name - pass it through unchanged.

        Args:
            voice: Voice name

        Returns:
            Voice name for OpenAI client
        """
        # If it's already an OpenAI voice name, return as-is
        if voice in OPENAI_VOICE_NAMES:
            return voice

        # Map "default" back to "alloy" for OpenAI compatibility
        if voice == "default":
            return "alloy"

        # Pass through real Kyutai voice names unchanged
        return voice

    def translate_client_message(self, message_json: str) -> Union[ora.ClientEvent, None]:
        """
        Translate OpenAI client message to internal format.

        Args:
            message_json: Raw JSON string from OpenAI client

        Returns:
            Internal ClientEvent object, or None if message should be ignored

        Raises:
            ValidationError: If message is invalid
        """
        try:
            # Parse JSON
            message_dict = json.loads(message_json)
            message_type = message_dict.get("type")

            # Handle session.create - OpenAI clients send this to initialize
            # Unmute doesn't have this event, so we convert it to session.update
            if message_type == "session.create":
                # Extract session config
                session = message_dict.get("session", {})

                # Map OpenAI voice to Kyutai voice
                if "voice" in session:
                    session["voice"] = self._map_openai_voice_to_kyutai(session["voice"])

                # Add allow_recording if not present (required by Unmute's SessionConfig)
                if "allow_recording" not in session:
                    session["allow_recording"] = False

                # Convert OpenAI string instructions to Unmute's ConstantInstructions format
                if "instructions" in session and isinstance(session["instructions"], str):
                    session["instructions"] = {
                        "type": "constant",
                        "text": session["instructions"]
                    }

                # Convert to session.update
                message_dict = {
                    "type": "session.update",
                    "session": session,
                }
                message_json = json.dumps(message_dict)
                self._session_configured = True

            # Handle response.create - OpenAI clients can manually trigger responses
            # This is used after providing function call results to continue the conversation
            elif message_type == "response.create":
                return self.client_event_adapter.validate_json(message_json)

            # Handle input_audio_buffer.commit - OpenAI manual mode
            # Unmute uses automatic mode (VAD-based), so we ignore this
            elif message_type == "input_audio_buffer.commit":
                return None

            # Handle input_audio_buffer.clear - OpenAI manual mode
            # Unmute doesn't support clearing the buffer (automatic mode)
            elif message_type == "input_audio_buffer.clear":
                return None

            # Handle conversation.item.create - for function call outputs
            elif message_type == "conversation.item.create":
                item = message_dict.get("item", {})
                if item.get("type") == "function_call_output":
                    # Pass through to handler for tool result processing
                    return self.client_event_adapter.validate_json(message_json)
                # Other item types not yet supported
                return None

            # Handle conversation item management (delete, truncate)
            # Unmute doesn't currently support these - ignore for now
            elif message_type in [
                "conversation.item.delete",
                "conversation.item.truncate",
            ]:
                # TODO: Support conversation item management
                return None

            # Handle session.update - map OpenAI voice to Kyutai voice and add required fields
            elif message_type == "session.update":
                session = message_dict.get("session", {})
                if "voice" in session:
                    session["voice"] = self._map_openai_voice_to_kyutai(session["voice"])
                # Add allow_recording if not present (required by Unmute's SessionConfig)
                if "allow_recording" not in session:
                    session["allow_recording"] = False
                # Convert OpenAI string instructions to Unmute's ConstantInstructions format
                if "instructions" in session and isinstance(session["instructions"], str):
                    session["instructions"] = {
                        "type": "constant",
                        "text": session["instructions"]
                    }
                message_dict["session"] = session
                message_json = json.dumps(message_dict)

            # For all other messages, validate against internal format
            return self.client_event_adapter.validate_json(message_json)

        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}")

    def translate_server_event(
        self,
        event: Union[ora.ServerEvent, Tuple[int, np.ndarray], AdditionalOutputs, CloseStream]
    ) -> Optional[str]:
        """
        Translate server event to OpenAI format.

        Args:
            event: Internal event object

        Returns:
            JSON string to send to OpenAI client, or None if nothing to send
        """
        # Handle different event types
        if isinstance(event, AdditionalOutputs):
            # OpenAI clients don't expect debug outputs - don't send
            return None

        elif isinstance(event, CloseStream):
            # Connection close - handled by WebSocket layer
            return None

        elif isinstance(event, ora.ServerEvent):
            # Filter out Unmute-specific events that OpenAI clients don't understand
            if isinstance(event, (
                ora.UnmuteAdditionalOutputs,
                ora.UnmuteResponseTextDeltaReady,
                ora.UnmuteResponseAudioDeltaReady,
                ora.UnmuteInterruptedByVAD,
            )):
                # Skip Unmute-specific events
                return None

            # Map Kyutai voice names back to OpenAI voice names
            event_dict = event.model_dump()

            if isinstance(event, ora.SessionUpdated):
                if "voice" in event_dict.get("session", {}):
                    event_dict["session"]["voice"] = self._map_kyutai_voice_to_openai(
                        event_dict["session"]["voice"]
                    )

            elif isinstance(event, ora.ResponseCreated):
                if "voice" in event_dict.get("response", {}):
                    event_dict["response"]["voice"] = self._map_kyutai_voice_to_openai(
                        event_dict["response"]["voice"]
                    )

            return json.dumps(event_dict)

        elif isinstance(event, tuple):
            # Audio tuple - should be handled by handle_audio_output
            raise ValueError("Audio tuples should be handled by handle_audio_output()")

        else:
            raise TypeError(f"Unknown event type: {type(event)}")

    @property
    def protocol_name(self) -> str:
        return "openai"

    @property
    def websocket_subprotocol(self) -> str:
        # OpenAI uses "realtime" subprotocol
        return "realtime"

    def get_session_config(self) -> dict:
        """Get OpenAI protocol session configuration."""
        return {
            "audio_codec": "pcm16",
            "sample_rate": self.sample_rate,
            "model": self.model,
            "bandwidth_multiplier": 1.0,  # PCM16 is the baseline
        }
