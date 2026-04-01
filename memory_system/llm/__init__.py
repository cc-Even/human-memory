"""
LLM模块
提供统一的LLM提供商接口和实现
"""

from .base import (
    LLMProvider,
    ChatMessage,
    ModelInfo,
    ChatResponse,
    EmbeddingResponse,
)
from .gemini_provider import GeminiProvider
from .dashscope_provider import DashScopeProvider
from .openai_provider import OpenAIProvider
from .factory import (
    LLMFactory,
    get_llm_factory,
    create_llm_provider,
)

__all__ = [
    # Base classes
    "LLMProvider",
    "ChatMessage",
    "ModelInfo",
    "ChatResponse",
    "EmbeddingResponse",
    # Providers
    "GeminiProvider",
    "DashScopeProvider",
    "OpenAIProvider",
    # Factory
    "LLMFactory",
    "get_llm_factory",
    "create_llm_provider",
]
