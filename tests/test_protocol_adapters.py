"""
Tests for protocol adapters.

Tests message translation, audio encoding/decoding, and voice mapping
for OpenAI and Native protocol adapters.
"""

import base64
import json

import numpy as np
import pytest

import unmute.openai_realtime_api_events as ora
from unmute.protocol.native import NativeProtocolAdapter
from unmute.protocol.openai import OpenAIProtocolAdapter


class TestNativeProtocolAdapter:
    """Tests for NativeProtocolAdapter."""

    @pytest.mark.asyncio
    async def test_setup_teardown(self):
        """Test adapter setup and teardown."""
        adapter = NativeProtocolAdapter(sample_rate=24000)
        await adapter.setup()
        assert adapter.opus_reader is not None
        assert adapter.opus_writer is not None

        await adapter.teardown()
        assert adapter.opus_reader is None
        assert adapter.opus_writer is None

    def test_protocol_name(self):
        """Test protocol name is correct."""
        adapter = NativeProtocolAdapter()
        assert adapter.protocol_name == "native"

    def test_websocket_subprotocol(self):
        """Test WebSocket subprotocol is correct."""
        adapter = NativeProtocolAdapter()
        assert adapter.websocket_subprotocol == "realtime"

    def test_get_session_config(self):
        """Test session config includes Opus codec info."""
        adapter = NativeProtocolAdapter()
        config = adapter.get_session_config()

        assert config["audio_codec"] == "opus"
        assert config["sample_rate"] == 24000
        assert config["bandwidth_multiplier"] == 0.06

    def test_translate_client_message_passthrough(self):
        """Test that client messages pass through unchanged."""
        adapter = NativeProtocolAdapter()

        # Create a session update message
        message = ora.SessionUpdate(
            session=ora.SessionConfig(
                instructions=None,
                voice="default",
                allow_recording=False,
            )
        )
        message_json = message.model_dump_json()

        # Translate
        translated = adapter.translate_client_message(message_json)

        # Should be the same
        assert isinstance(translated, ora.SessionUpdate)
        assert translated.session.voice == "default"

    def test_translate_server_event_passthrough(self):
        """Test that server events pass through with JSON serialization."""
        adapter = NativeProtocolAdapter()

        # Create a server event
        event = ora.ResponseTextDelta(delta="Hello, ")

        # Translate
        translated_json = adapter.translate_server_event(event)

        # Should be JSON
        translated_dict = json.loads(translated_json)
        assert translated_dict["type"] == "response.text.delta"
        assert translated_dict["delta"] == "Hello, "

    def test_translate_server_event_additional_outputs(self):
        """Test that AdditionalOutputs is wrapped correctly."""
        adapter = NativeProtocolAdapter()
        from fastrtc import AdditionalOutputs

        # Create additional outputs - AdditionalOutputs takes *args
        outputs = AdditionalOutputs({"debug": "info"})

        # Translate
        translated_json = adapter.translate_server_event(outputs)

        # Should be wrapped in UnmuteAdditionalOutputs
        translated_dict = json.loads(translated_json)
        assert translated_dict["type"] == "unmute.additional_outputs"
        assert "args" in translated_dict


class TestOpenAIProtocolAdapter:
    """Tests for OpenAIProtocolAdapter."""

    @pytest.mark.asyncio
    async def test_setup_teardown(self):
        """Test adapter setup and teardown."""
        adapter = OpenAIProtocolAdapter(sample_rate=24000)
        await adapter.setup()
        # OpenAI adapter doesn't need codec setup
        await adapter.teardown()

    def test_protocol_name(self):
        """Test protocol name is correct."""
        adapter = OpenAIProtocolAdapter()
        assert adapter.protocol_name == "openai"

    def test_websocket_subprotocol(self):
        """Test WebSocket subprotocol is correct."""
        adapter = OpenAIProtocolAdapter()
        assert adapter.websocket_subprotocol == "realtime"

    def test_get_session_config(self):
        """Test session config includes PCM16 codec info."""
        adapter = OpenAIProtocolAdapter()
        config = adapter.get_session_config()

        assert config["audio_codec"] == "pcm16"
        assert config["sample_rate"] == 24000
        assert config["model"] == "gpt-4o-realtime-preview"
        assert config["bandwidth_multiplier"] == 1.0

    def test_voice_mapping_openai_to_kyutai(self):
        """Test OpenAI voice names map to Kyutai voices."""
        adapter = OpenAIProtocolAdapter()

        # Standard OpenAI voices map to "default"
        assert adapter._map_openai_voice_to_kyutai("alloy") == "default"
        assert adapter._map_openai_voice_to_kyutai("echo") == "default"
        assert adapter._map_openai_voice_to_kyutai("shimmer") == "default"
        # Non-OpenAI voice names (real Kyutai voices) pass through unchanged
        assert adapter._map_openai_voice_to_kyutai("some_kyutai_voice") == "some_kyutai_voice"

    def test_voice_mapping_kyutai_to_openai(self):
        """Test Kyutai voice names map back for OpenAI clients."""
        adapter = OpenAIProtocolAdapter()

        # OpenAI voices should pass through as-is
        assert adapter._map_kyutai_voice_to_openai("alloy") == "alloy"

        # "default" maps to "alloy" for OpenAI compatibility
        assert adapter._map_kyutai_voice_to_openai("default") == "alloy"
        # Real Kyutai voice names pass through unchanged
        assert adapter._map_kyutai_voice_to_openai("custom_voice") == "custom_voice"

    def test_translate_session_create_to_update(self):
        """Test that session.create is converted to session.update."""
        adapter = OpenAIProtocolAdapter()

        # OpenAI clients send session.create with simple voice setting
        message_dict = {
            "type": "session.create",
            "session": {
                "voice": "echo",
                "allow_recording": False,
            },
        }
        message_json = json.dumps(message_dict)

        # Translate
        translated = adapter.translate_client_message(message_json)

        # Should be converted to session.update with voice mapping
        assert isinstance(translated, ora.SessionUpdate)
        assert translated.session.voice == "default"  # echo → default

    def test_translate_session_update_voice_mapping(self):
        """Test that session.update maps voice names."""
        adapter = OpenAIProtocolAdapter()

        message_dict = {
            "type": "session.update",
            "session": {
                "voice": "shimmer",
                "allow_recording": False,
            },
        }
        message_json = json.dumps(message_dict)

        # Translate
        translated = adapter.translate_client_message(message_json)

        # Voice should be mapped
        assert isinstance(translated, ora.SessionUpdate)
        assert translated.session.voice == "default"

    def test_translate_response_create_passed_through(self):
        """Test that response.create is passed through (for tool calling support)."""
        adapter = OpenAIProtocolAdapter()

        message_dict = {
            "type": "response.create",
        }
        message_json = json.dumps(message_dict)

        # Translate
        translated = adapter.translate_client_message(message_json)

        # Should be passed through for tool calling support
        assert translated is not None
        assert isinstance(translated, ora.ResponseCreate)

    def test_translate_input_audio_buffer_commit_filtered(self):
        """Test that input_audio_buffer.commit is filtered out."""
        adapter = OpenAIProtocolAdapter()

        message_dict = {
            "type": "input_audio_buffer.commit",
        }
        message_json = json.dumps(message_dict)

        # Translate
        translated = adapter.translate_client_message(message_json)

        # Should be filtered (None)
        assert translated is None

    def test_translate_server_event_filters_unmute_extensions(self):
        """Test that Unmute-specific events are filtered out for OpenAI clients."""
        adapter = OpenAIProtocolAdapter()

        # Create Unmute-specific events
        unmute_events = [
            ora.UnmuteAdditionalOutputs(args={"test": "data"}),
            ora.UnmuteResponseTextDeltaReady(delta="ready"),
            ora.UnmuteResponseAudioDeltaReady(number_of_samples=100),
            ora.UnmuteInterruptedByVAD(),
        ]

        for event in unmute_events:
            translated = adapter.translate_server_event(event)
            # Should all be filtered (None)
            assert translated is None, f"Event {event.type} should be filtered"

    def test_translate_server_event_passes_standard_events(self):
        """Test that standard events pass through."""
        adapter = OpenAIProtocolAdapter()

        # Create standard events
        event = ora.ResponseTextDelta(delta="Hello")

        # Translate
        translated_json = adapter.translate_server_event(event)

        # Should pass through
        assert translated_json is not None
        translated_dict = json.loads(translated_json)
        assert translated_dict["type"] == "response.text.delta"
        assert translated_dict["delta"] == "Hello"

    def test_translate_session_updated_voice_mapping(self):
        """Test that session.updated maps voice back to OpenAI names."""
        adapter = OpenAIProtocolAdapter()

        # Create session.updated with Kyutai voice
        event = ora.SessionUpdated(
            session=ora.SessionConfig(
                instructions=None,
                voice="default",
                allow_recording=False,
            )
        )

        # Translate
        translated_json = adapter.translate_server_event(event)
        translated_dict = json.loads(translated_json)

        # Voice should be mapped back to OpenAI name
        assert translated_dict["session"]["voice"] == "alloy"

    def test_translate_response_created_voice_mapping(self):
        """Test that response.created maps voice back to OpenAI names."""
        adapter = OpenAIProtocolAdapter()

        # Create response.created with Kyutai voice
        event = ora.ResponseCreated(
            response=ora.Response(
                status="in_progress",
                voice="default",
            )
        )

        # Translate
        translated_json = adapter.translate_server_event(event)
        translated_dict = json.loads(translated_json)

        # Voice should be mapped back
        assert translated_dict["response"]["voice"] == "alloy"

    @pytest.mark.asyncio
    async def test_decode_audio_pcm16(self):
        """Test decoding PCM16 audio."""
        adapter = OpenAIProtocolAdapter()

        # Create some PCM16 audio
        pcm16 = np.array([0, 100, -100, 32767, -32767], dtype=np.int16)
        pcm16_bytes = pcm16.tobytes()
        base64_audio = base64.b64encode(pcm16_bytes).decode("utf-8")

        # Decode
        pcm_float32 = await adapter.decode_audio(base64_audio)

        # Verify
        assert pcm_float32 is not None
        assert pcm_float32.dtype == np.float32
        assert len(pcm_float32) == 5

    @pytest.mark.asyncio
    async def test_encode_audio_pcm16(self):
        """Test encoding PCM16 audio."""
        adapter = OpenAIProtocolAdapter()

        # Create some float32 audio
        pcm_float32 = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

        # Encode
        base64_audio = await adapter.encode_audio(pcm_float32)

        # Verify
        assert base64_audio is not None
        assert isinstance(base64_audio, str)

        # Decode to verify
        pcm16_bytes = base64.b64decode(base64_audio)
        pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        assert len(pcm16) == 5

    @pytest.mark.asyncio
    async def test_audio_roundtrip(self):
        """Test that audio encode/decode roundtrip works."""
        adapter = OpenAIProtocolAdapter()

        # Original float32 audio
        original = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

        # Encode to PCM16
        encoded = await adapter.encode_audio(original)

        # Decode back to float32
        decoded = await adapter.decode_audio(encoded)

        # Should be very close
        assert np.allclose(decoded, original, atol=0.001)

    @pytest.mark.asyncio
    async def test_handle_audio_message(self):
        """Test handling InputAudioBufferAppend message."""
        adapter = OpenAIProtocolAdapter()

        # Create a message with PCM16 audio
        pcm16 = np.array([0, 100, -100], dtype=np.int16)
        pcm16_bytes = pcm16.tobytes()
        base64_audio = base64.b64encode(pcm16_bytes).decode("utf-8")

        message = ora.InputAudioBufferAppend(audio=base64_audio)

        # Handle the message
        result = await adapter.handle_audio_message(message)

        # Should return (sample_rate, audio_array) tuple
        assert result is not None
        sample_rate, audio_array = result

        assert sample_rate == 24000
        assert audio_array.shape[0] == 1  # Batch dimension
        assert audio_array.shape[1] == 3  # 3 samples

    @pytest.mark.asyncio
    async def test_handle_audio_output(self):
        """Test handling audio output tuple."""
        adapter = OpenAIProtocolAdapter()

        # Create audio output tuple (as from UnmuteHandler)
        sample_rate = 24000
        audio = np.array([[0.0, 0.5, -0.5]], dtype=np.float32)  # 2D with batch dim
        audio_tuple = (sample_rate, audio)

        # Handle the output
        result = await adapter.handle_audio_output(audio_tuple)

        # Should return ResponseAudioDelta
        assert result is not None
        assert isinstance(result, ora.ResponseAudioDelta)
        assert result.delta is not None
        assert isinstance(result.delta, str)

        # Verify it's valid base64 PCM16
        pcm16_bytes = base64.b64decode(result.delta)
        pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        assert len(pcm16) == 3


class TestProtocolComparison:
    """Tests comparing Native and OpenAI protocols."""

    @pytest.mark.asyncio
    async def test_both_adapters_handle_same_internal_message(self):
        """Test that both adapters can handle the same internal message format."""
        native = NativeProtocolAdapter()
        openai = OpenAIProtocolAdapter()

        # Create an internal server event
        event = ora.ResponseTextDelta(delta="Test message")

        # Both should be able to translate it
        native_json = native.translate_server_event(event)
        openai_json = openai.translate_server_event(event)

        # Both should produce valid JSON
        native_dict = json.loads(native_json)
        openai_dict = json.loads(openai_json)

        # Both should have the same type and delta
        assert native_dict["type"] == openai_dict["type"]
        assert native_dict["delta"] == openai_dict["delta"]

    def test_bandwidth_difference(self):
        """Test that bandwidth multipliers reflect the difference."""
        native = NativeProtocolAdapter()
        openai = OpenAIProtocolAdapter()

        native_config = native.get_session_config()
        openai_config = openai.get_session_config()

        # Native (Opus) should use ~6% of PCM16 bandwidth
        assert native_config["bandwidth_multiplier"] < openai_config["bandwidth_multiplier"]
        assert native_config["bandwidth_multiplier"] == 0.06
        assert openai_config["bandwidth_multiplier"] == 1.0
