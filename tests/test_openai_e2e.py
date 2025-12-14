"""
End-to-end test for OpenAI protocol using websockets.

This test connects to a running unmute server and verifies that
the OpenAI protocol endpoint works correctly.

To run this test, you need:
1. unmute server running on localhost:8000
2. All services (STT, TTS, LLM) available

Run with: pytest tests/test_openai_e2e.py -v

To skip if server not running: pytest tests/test_openai_e2e.py -v -m "not requires_server"

Tests mirror the HTML client behavior (openai-realtime-client.html):
- session.update with string instructions
- session.update with voice selection
- input_audio_buffer.append with PCM16 audio
- Receiving response.text.done, response.audio.delta, etc.
"""

import asyncio
import base64
import json
from typing import Any

import numpy as np
import pytest
import websockets
from websockets.exceptions import WebSocketException

# Mark all tests in this file as requiring a running server
pytestmark = pytest.mark.requires_server

# Known OpenAI-compatible message types that should be handled
OPENAI_MESSAGE_TYPES = {
    # Client -> Server
    "session.update",
    "session.create",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "input_audio_buffer.clear",
    "response.create",
    "conversation.item.create",
    "conversation.item.delete",
    "conversation.item.truncate",
    # Server -> Client
    "session.updated",
    "session.created",
    "response.created",
    "response.text.delta",
    "response.text.done",
    "response.audio.delta",
    "response.audio.done",
    "response.done",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "conversation.item.input_audio_transcription.delta",
    "error",
}


def encode_pcm16_audio(float32_array: np.ndarray) -> str:
    """Encode float32 audio to base64 PCM16 (mimics HTML client's encodePCM16)."""
    # Clip to [-1, 1]
    float32_array = np.clip(float32_array, -1.0, 1.0)
    # Convert to int16
    int16_array = (float32_array * 32767).astype(np.int16)
    # Encode to base64
    return base64.b64encode(int16_array.tobytes()).decode("utf-8")


def decode_pcm16_audio(base64_audio: str) -> np.ndarray:
    """Decode base64 PCM16 to float32 audio (mimics HTML client's decodePCM16)."""
    pcm16_bytes = base64.b64decode(base64_audio)
    int16_array = np.frombuffer(pcm16_bytes, dtype=np.int16)
    # Convert to float32 [-1, 1]
    return int16_array.astype(np.float32) / 32768.0


async def receive_with_timeout(ws, timeout: float = 5.0) -> dict[str, Any]:
    """Receive a message with timeout, parse JSON."""
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(raw)


class TestOpenAIProtocolE2E:
    """End-to-end tests for OpenAI protocol."""

    @pytest.fixture
    async def ws_connection(self):
        """Create a WebSocket connection to the OpenAI endpoint."""
        uri = "ws://localhost:8000/v1/realtime/openai?model=gpt-4o-realtime-preview"

        try:
            async with websockets.connect(
                uri,
                subprotocols=["realtime"],
                open_timeout=5,
            ) as websocket:
                yield websocket
        except (ConnectionRefusedError, WebSocketException, OSError) as e:
            pytest.skip(f"unmute server not running: {e}")

    @pytest.mark.asyncio
    async def test_connection_established(self, ws_connection):
        """Test that we can establish a connection to the OpenAI endpoint."""
        # If we got here, connection was successful
        assert ws_connection.open
        assert ws_connection.subprotocol == "realtime"

    @pytest.mark.asyncio
    async def test_session_update(self, ws_connection):
        """Test sending a session.update message."""
        # Send session.update
        message = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": "You are a helpful assistant.",
            },
        }

        await ws_connection.send(json.dumps(message))

        # Should receive session.updated back
        response_raw = await asyncio.wait_for(ws_connection.recv(), timeout=5)
        response = json.loads(response_raw)

        assert response["type"] == "session.updated"
        assert "session" in response
        # Voice should be mapped (alloy → default → alloy in response)
        assert response["session"]["voice"] == "alloy"

    @pytest.mark.asyncio
    async def test_session_update_mimics_html_client(self, ws_connection):
        """Test session.update exactly as HTML client sends it (openai-realtime-client.html line 420-427)."""
        # This is exactly what the HTML client sends on connect
        session_update = {
            "type": "session.update",
            "session": {
                "voice": "alloy",  # From voiceSelect.value
                "instructions": "You are a helpful assistant.",  # From instructionsInput.value
            },
        }

        await ws_connection.send(json.dumps(session_update))

        # Should receive session.updated
        response = await receive_with_timeout(ws_connection)

        assert response["type"] == "session.updated", f"Expected session.updated, got {response}"
        assert "session" in response

    @pytest.mark.asyncio
    async def test_audio_buffer_append(self, ws_connection):
        """Test sending audio via input_audio_buffer.append."""
        # Create some PCM16 audio (silence)
        pcm16 = np.zeros(4800, dtype=np.int16)  # 200ms at 24kHz
        pcm16_bytes = pcm16.tobytes()
        base64_audio = base64.b64encode(pcm16_bytes).decode("utf-8")

        # Send audio
        message = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio,
        }

        await ws_connection.send(json.dumps(message))

        # Server should accept it (no error)
        # We might not get an immediate response for silence
        await asyncio.sleep(0.1)

        # Check connection still open
        assert ws_connection.open

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, ws_connection):
        """Test that invalid JSON returns an error."""
        # Send invalid JSON
        await ws_connection.send("not valid json{")

        # Should receive an error
        response_raw = await asyncio.wait_for(ws_connection.recv(), timeout=5)
        response = json.loads(response_raw)

        assert response["type"] == "error"
        assert "error" in response
        assert "Invalid JSON" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_session_create_converted_to_update(self, ws_connection):
        """Test that session.create is converted to session.update."""
        # OpenAI clients might send session.create
        message = {
            "type": "session.create",
            "session": {
                "voice": "echo",
            },
        }

        await ws_connection.send(json.dumps(message))

        # Should receive session.updated (not session.created)
        response_raw = await asyncio.wait_for(ws_connection.recv(), timeout=5)
        response = json.loads(response_raw)

        assert response["type"] == "session.updated"
        # Voice should be mapped (echo → default → alloy)
        assert response["session"]["voice"] == "alloy"


class TestOpenAIProtocolCompatibility:
    """Test compatibility with OpenAI SDK message formats.

    These tests don't require a running server - they test the protocol adapter directly.
    """

    @pytest.mark.asyncio
    async def test_openai_sdk_message_format(self):
        """Test that messages match OpenAI SDK format."""
        # Example OpenAI SDK session.update
        openai_message = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "You are a helpful assistant.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.8,
            },
        }

        # Verify our protocol adapter can handle this
        from unmute.protocol.openai import OpenAIProtocolAdapter

        adapter = OpenAIProtocolAdapter()
        message_json = json.dumps(openai_message)

        # Should translate without error (even if some fields are ignored)
        translated = adapter.translate_client_message(message_json)

        assert translated is not None
        assert hasattr(translated, "type")

    def test_html_client_session_update_format(self):
        """Test the exact format sent by openai-realtime-client.html."""
        from unmute.protocol.openai import OpenAIProtocolAdapter

        adapter = OpenAIProtocolAdapter()

        # Exactly what the HTML client sends (line 420-427)
        html_client_message = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": "You are a helpful assistant.",
            },
        }

        translated = adapter.translate_client_message(json.dumps(html_client_message))

        assert translated is not None
        assert translated.type == "session.update"
        # Instructions should be converted to ConstantInstructions
        assert translated.session.instructions is not None
        assert translated.session.instructions.type == "constant"
        assert translated.session.instructions.text == "You are a helpful assistant."
        # allow_recording should be added
        assert translated.session.allow_recording == False

    def test_html_client_audio_format(self):
        """Test audio encoding/decoding matches HTML client."""
        from unmute.protocol.openai import OpenAIProtocolAdapter

        adapter = OpenAIProtocolAdapter()

        # Create audio like HTML client does
        float32_audio = np.sin(np.linspace(0, 2 * np.pi * 440, 4096)).astype(np.float32)
        base64_audio = encode_pcm16_audio(float32_audio)

        # Create message like HTML client (line 540-543)
        message = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio,
        }

        translated = adapter.translate_client_message(json.dumps(message))

        assert translated is not None
        assert translated.type == "input_audio_buffer.append"
        assert translated.audio == base64_audio

    def test_all_openai_voice_names_passthrough(self):
        """Test that all OpenAI voice names pass through unchanged."""
        from unmute.protocol.openai import OpenAIProtocolAdapter, OPENAI_VOICE_NAMES

        adapter = OpenAIProtocolAdapter()

        for voice in OPENAI_VOICE_NAMES:
            message = {
                "type": "session.update",
                "session": {
                    "voice": voice,
                    "allow_recording": False,
                },
            }

            translated = adapter.translate_client_message(json.dumps(message))
            assert translated is not None, f"Failed for voice: {voice}"
            # Voice should pass through unchanged (no mapping)
            assert translated.session.voice == voice, f"Voice '{voice}' should pass through unchanged"


def create_pcm16_sine_wave(frequency: float = 440.0, duration: float = 0.1, sample_rate: int = 24000) -> str:
    """Helper to create a base64-encoded PCM16 sine wave."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    sine_wave = np.sin(2.0 * np.pi * frequency * t)

    # Scale to int16 range
    pcm16 = (sine_wave * 32767).astype(np.int16)
    pcm16_bytes = pcm16.tobytes()

    return base64.b64encode(pcm16_bytes).decode("utf-8")


class TestOpenAIProtocolAudio:
    """Test audio streaming with OpenAI protocol."""

    @pytest.fixture
    async def ws_connection(self):
        """Create a WebSocket connection to the OpenAI endpoint."""
        uri = "ws://localhost:8000/v1/realtime/openai"

        try:
            async with websockets.connect(uri, subprotocols=["realtime"]) as websocket:
                yield websocket
        except (ConnectionRefusedError, WebSocketException, OSError) as e:
            pytest.skip(f"unmute server not running: {e}")

    @pytest.mark.asyncio
    async def test_send_sine_wave_audio(self, ws_connection):
        """Test sending a sine wave as audio."""
        # Create a sine wave (A4 note, 440 Hz)
        base64_audio = create_pcm16_sine_wave(frequency=440.0, duration=0.5)

        # Send it
        message = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio,
        }

        await ws_connection.send(json.dumps(message))

        # Server should accept it
        await asyncio.sleep(0.1)
        assert ws_connection.open

    @pytest.mark.asyncio
    async def test_multiple_audio_chunks(self, ws_connection):
        """Test sending multiple audio chunks."""
        # Send multiple chunks of audio
        for i in range(5):
            base64_audio = create_pcm16_sine_wave(duration=0.1)

            message = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio,
            }

            await ws_connection.send(json.dumps(message))
            await asyncio.sleep(0.05)

        # Server should still be connected
        assert ws_connection.open


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    import sys

    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
