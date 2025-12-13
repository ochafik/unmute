"""
Audio transcoding utilities for protocol adapters.

Provides functions to convert between different audio formats:
- PCM16 (16-bit signed integer, used by OpenAI Realtime API)
- PCM Float32 (32-bit float, used internally by Unmute)
- Opus (compressed codec, used by native Unmute protocol)
"""

import base64

import numpy as np


def pcm16_to_float32(pcm16_bytes: bytes) -> np.ndarray:
    """
    Convert PCM16 audio bytes to Float32 numpy array.

    Args:
        pcm16_bytes: Raw PCM16 bytes (16-bit signed little-endian integers)

    Returns:
        Float32 numpy array with values in range [-1.0, 1.0]
    """
    # Decode PCM16 bytes to int16 array
    pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)

    # Convert to float32 in range [-1.0, 1.0]
    pcm_float32 = pcm16.astype(np.float32) / 32768.0

    return pcm_float32


def float32_to_pcm16(pcm_float32: np.ndarray) -> bytes:
    """
    Convert Float32 numpy array to PCM16 audio bytes.

    Args:
        pcm_float32: Float32 numpy array with values in range [-1.0, 1.0]

    Returns:
        Raw PCM16 bytes (16-bit signed little-endian integers)
    """
    # Clip to valid range [-1.0, 1.0]
    pcm_float32 = np.clip(pcm_float32, -1.0, 1.0)

    # Convert to int16 (scale by 32767 and round)
    pcm16 = (pcm_float32 * 32767.0).astype(np.int16)

    # Convert to bytes
    return pcm16.tobytes()


def decode_pcm16_base64(base64_audio: str) -> np.ndarray:
    """
    Decode base64-encoded PCM16 audio to Float32 numpy array.

    Args:
        base64_audio: Base64-encoded PCM16 audio

    Returns:
        Float32 numpy array with values in range [-1.0, 1.0]
    """
    pcm16_bytes = base64.b64decode(base64_audio)
    return pcm16_to_float32(pcm16_bytes)


def encode_pcm16_base64(pcm_float32: np.ndarray) -> str:
    """
    Encode Float32 numpy array to base64-encoded PCM16 audio.

    Args:
        pcm_float32: Float32 numpy array with values in range [-1.0, 1.0]

    Returns:
        Base64-encoded PCM16 audio
    """
    pcm16_bytes = float32_to_pcm16(pcm_float32)
    return base64.b64encode(pcm16_bytes).decode("utf-8")
