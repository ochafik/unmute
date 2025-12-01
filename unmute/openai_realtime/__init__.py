"""OpenAI Realtime API integration for Unmute.

This module provides a handler that connects to OpenAI's Realtime API,
allowing direct voice-to-voice conversations with GPT-4o models.

Architecture:
    Browser <--PCM16/WebSocket--> UnmuteBackend <--PCM16/WebSocket--> OpenAI Realtime API

The module is designed to minimize changes to existing Unmute code:
- New handler (OpenAIRealtimeHandler) as alternative to UnmuteHandler
- Route selection based on session configuration
- Native PCM16 audio format for optimal latency (no conversion)
"""

from unmute.openai_realtime.handler import OpenAIRealtimeHandler
from unmute.openai_realtime.client import OpenAIRealtimeClient

__all__ = ["OpenAIRealtimeHandler", "OpenAIRealtimeClient"]
