"""
Tests for audio transcoding utilities.

Tests conversion between PCM16 and Float32 audio formats.
"""

import base64

import numpy as np
import pytest

from unmute.protocol.audio_transcoding import (
    decode_pcm16_base64,
    encode_pcm16_base64,
    float32_to_pcm16,
    pcm16_to_float32,
)


def test_pcm16_to_float32_zero():
    """Test that zero PCM16 converts to zero float32."""
    pcm16_bytes = np.zeros(100, dtype=np.int16).tobytes()
    pcm_float32 = pcm16_to_float32(pcm16_bytes)

    assert pcm_float32.dtype == np.float32
    assert len(pcm_float32) == 100
    assert np.allclose(pcm_float32, 0.0)


def test_pcm16_to_float32_max():
    """Test that max PCM16 value converts correctly."""
    # Max positive value in int16
    pcm16_bytes = np.array([32767], dtype=np.int16).tobytes()
    pcm_float32 = pcm16_to_float32(pcm16_bytes)

    # Should be very close to 1.0 (32767 / 32768)
    assert np.allclose(pcm_float32, 1.0, atol=0.01)


def test_pcm16_to_float32_min():
    """Test that min PCM16 value converts correctly."""
    # Max negative value in int16
    pcm16_bytes = np.array([-32768], dtype=np.int16).tobytes()
    pcm_float32 = pcm16_to_float32(pcm16_bytes)

    # Should be -1.0
    assert np.allclose(pcm_float32, -1.0)


def test_pcm16_to_float32_range():
    """Test conversion of a range of values."""
    pcm16 = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
    pcm16_bytes = pcm16.tobytes()
    pcm_float32 = pcm16_to_float32(pcm16_bytes)

    expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    assert np.allclose(pcm_float32, expected, atol=0.01)


def test_float32_to_pcm16_zero():
    """Test that zero float32 converts to zero PCM16."""
    pcm_float32 = np.zeros(100, dtype=np.float32)
    pcm16_bytes = float32_to_pcm16(pcm_float32)

    pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    assert len(pcm16) == 100
    assert np.all(pcm16 == 0)


def test_float32_to_pcm16_max():
    """Test that max float32 value converts correctly."""
    pcm_float32 = np.array([1.0], dtype=np.float32)
    pcm16_bytes = float32_to_pcm16(pcm_float32)

    pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    # Should be 32767 (max int16)
    assert pcm16[0] == 32767


def test_float32_to_pcm16_min():
    """Test that min float32 value converts correctly."""
    pcm_float32 = np.array([-1.0], dtype=np.float32)
    pcm16_bytes = float32_to_pcm16(pcm_float32)

    pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    # Should be -32767 (close to min int16)
    assert pcm16[0] == -32767


def test_float32_to_pcm16_clipping():
    """Test that values outside [-1, 1] are clipped."""
    pcm_float32 = np.array([-2.0, -1.5, 1.5, 2.0], dtype=np.float32)
    pcm16_bytes = float32_to_pcm16(pcm_float32)

    pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    # All should be clipped to [-32767, 32767]
    assert pcm16[0] == -32767  # -2.0 clipped to -1.0
    assert pcm16[1] == -32767  # -1.5 clipped to -1.0
    assert pcm16[2] == 32767   # 1.5 clipped to 1.0
    assert pcm16[3] == 32767   # 2.0 clipped to 1.0


def test_float32_to_pcm16_range():
    """Test conversion of a range of values."""
    pcm_float32 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    pcm16_bytes = float32_to_pcm16(pcm_float32)

    pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    expected = np.array([-32767, -16384, 0, 16384, 32767], dtype=np.int16)
    # Allow small tolerance for rounding
    assert np.allclose(pcm16, expected, atol=1)


def test_roundtrip_pcm16_float32_pcm16():
    """Test that PCM16 → Float32 → PCM16 roundtrip is lossless (within tolerance)."""
    original_pcm16 = np.array([-32767, -16384, 0, 16384, 32767], dtype=np.int16)
    pcm16_bytes = original_pcm16.tobytes()

    # Convert to float32
    pcm_float32 = pcm16_to_float32(pcm16_bytes)

    # Convert back to PCM16
    roundtrip_bytes = float32_to_pcm16(pcm_float32)
    roundtrip_pcm16 = np.frombuffer(roundtrip_bytes, dtype=np.int16)

    # Should be very close (within 1 due to rounding)
    assert np.allclose(roundtrip_pcm16, original_pcm16, atol=1)


def test_roundtrip_float32_pcm16_float32():
    """Test that Float32 → PCM16 → Float32 roundtrip preserves values (within tolerance)."""
    original_float32 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

    # Convert to PCM16
    pcm16_bytes = float32_to_pcm16(original_float32)

    # Convert back to float32
    roundtrip_float32 = pcm16_to_float32(pcm16_bytes)

    # Should be very close (within small tolerance for quantization)
    assert np.allclose(roundtrip_float32, original_float32, atol=0.001)


def test_decode_pcm16_base64():
    """Test decoding base64-encoded PCM16."""
    # Create some PCM16 data
    pcm16 = np.array([0, 100, -100, 32767, -32767], dtype=np.int16)
    pcm16_bytes = pcm16.tobytes()
    base64_audio = base64.b64encode(pcm16_bytes).decode("utf-8")

    # Decode it
    pcm_float32 = decode_pcm16_base64(base64_audio)

    # Verify
    assert pcm_float32.dtype == np.float32
    assert len(pcm_float32) == 5
    assert np.allclose(pcm_float32[0], 0.0)
    assert np.allclose(pcm_float32[4], -1.0, atol=0.01)


def test_encode_pcm16_base64():
    """Test encoding float32 to base64-encoded PCM16."""
    pcm_float32 = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

    # Encode it
    base64_audio = encode_pcm16_base64(pcm_float32)

    # Verify it's valid base64
    assert isinstance(base64_audio, str)
    pcm16_bytes = base64.b64decode(base64_audio)

    # Decode to PCM16 and verify
    pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    assert len(pcm16) == 5
    assert pcm16[0] == 0
    assert np.allclose(pcm16[3], 32767, atol=1)


def test_base64_roundtrip():
    """Test that base64 encode/decode roundtrip is lossless."""
    original_float32 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

    # Encode to base64 PCM16
    base64_audio = encode_pcm16_base64(original_float32)

    # Decode back to float32
    roundtrip_float32 = decode_pcm16_base64(base64_audio)

    # Should be very close
    assert np.allclose(roundtrip_float32, original_float32, atol=0.001)


def test_empty_array():
    """Test handling of empty arrays."""
    # Empty float32
    empty_float32 = np.array([], dtype=np.float32)
    pcm16_bytes = float32_to_pcm16(empty_float32)
    assert len(pcm16_bytes) == 0

    # Empty PCM16
    empty_pcm16 = np.array([], dtype=np.int16).tobytes()
    pcm_float32 = pcm16_to_float32(empty_pcm16)
    assert len(pcm_float32) == 0


def test_large_array():
    """Test handling of large arrays (simulating real audio)."""
    # Simulate 1 second of audio at 24kHz
    num_samples = 24000
    pcm_float32 = np.random.uniform(-0.5, 0.5, num_samples).astype(np.float32)

    # Encode to PCM16
    pcm16_bytes = float32_to_pcm16(pcm_float32)

    # Verify size
    assert len(pcm16_bytes) == num_samples * 2  # 2 bytes per sample

    # Decode back
    roundtrip_float32 = pcm16_to_float32(pcm16_bytes)

    # Should be close (within quantization tolerance)
    assert len(roundtrip_float32) == num_samples
    assert np.allclose(roundtrip_float32, pcm_float32, atol=0.001)


def test_sine_wave():
    """Test with a sine wave (realistic audio signal)."""
    # Generate a sine wave
    sample_rate = 24000
    duration = 0.1  # 100ms
    frequency = 440.0  # A4 note

    t = np.arange(0, duration, 1.0 / sample_rate)
    sine_wave = np.sin(2.0 * np.pi * frequency * t).astype(np.float32)

    # Encode to PCM16
    pcm16_bytes = float32_to_pcm16(sine_wave)

    # Decode back
    roundtrip = pcm16_to_float32(pcm16_bytes)

    # Should be very close
    assert np.allclose(roundtrip, sine_wave, atol=0.001)

    # Verify base64 roundtrip too
    base64_audio = encode_pcm16_base64(sine_wave)
    base64_roundtrip = decode_pcm16_base64(base64_audio)
    assert np.allclose(base64_roundtrip, sine_wave, atol=0.001)
