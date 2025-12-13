"""
Protocol adapter module for multi-protocol support in Unmute.

This module provides a protocol adapter pattern that allows Unmute to support
multiple real-time communication protocols:
- Native Unmute protocol (Opus-encoded audio)
- OpenAI Realtime API protocol (PCM16-encoded audio)
- RTVI protocol (PCM Float32-encoded audio)

The adapter pattern keeps UnmuteHandler protocol-agnostic while translating
between external protocol formats and Unmute's internal representation.
"""

from unmute.protocol.base import ProtocolAdapter
from unmute.protocol.native import NativeProtocolAdapter
from unmute.protocol.openai import OpenAIProtocolAdapter

__all__ = [
    "ProtocolAdapter",
    "NativeProtocolAdapter",
    "OpenAIProtocolAdapter",
]
