"""
LLM提供商工厂
根据配置动态创建LLM提供商实例
"""

from typing import Optional, Dict, Any
import logging

from .base import LLMProvider, ChatMessage
from .gemini_provider import GeminiProvider
from .dashscope_provider import DashScopeProvider
from .openai_provider import OpenAIProvider
from memory_system.utils.logger import get_llm_logger


class LLMFactory:
    """
    LLM提供商工厂类

    根据配置动态创建具体的LLM提供商实例
    """

    # 支持的提供商类型
    SUPPORTED_PROVIDERS = ["gemini", "dashscope", "openai"]

    # 单例实例
    _instance: Optional["LLMFactory"] = None
    _providers: Dict[str, LLMProvider] = {}

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化工厂"""
        self.logger = get_llm_logger()

    def create_provider(
        self,
        provider_name: str,
        api_key: str,
        **kwargs
    ) -> LLMProvider:
        """
        创建LLM提供商实例

        Args:
            provider_name: 提供商名称 (gemini/dashscope)
            api_key: API密钥
            **kwargs: 其他配置参数

        Returns:
            LLMProvider: 提供商实例

        Raises:
            ValueError: 不支持的提供商类型
        """
        provider_name = provider_name.lower()

        if provider_name == "gemini":
            return GeminiProvider(api_key, **kwargs)
        elif provider_name == "dashscope":
            return DashScopeProvider(api_key, **kwargs)
        elif provider_name == "openai":
            return OpenAIProvider(api_key, **kwargs)
        else:
            raise ValueError(
                f"不支持的LLM提供商: {provider_name}。"
                f"支持的提供商: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )

    def get_or_create_provider(
        self,
        provider_name: str,
        api_key: str,
        **kwargs
    ) -> LLMProvider:
        """
        获取或创建LLM提供商实例（缓存）

        Args:
            provider_name: 提供商名称
            api_key: API密钥
            **kwargs: 其他配置参数

        Returns:
            LLMProvider: 提供商实例
        """
        provider_name = provider_name.lower()
        cache_key = f"{provider_name}_{hash(api_key)}"

        # 检查缓存
        if cache_key in self._providers:
            self.logger.info(f"从缓存获取LLM提供商: {provider_name}")
            return self._providers[cache_key]

        # 创建新实例
        provider = self.create_provider(provider_name, api_key, **kwargs)
        self._providers[cache_key] = provider

        return provider

    def create_chat_provider(
        self,
        provider_name: str,
        api_key: str,
        **kwargs
    ) -> LLMProvider:
        """
        创建用于对话的LLM提供商

        Args:
            provider_name: 提供商名称
            api_key: API密钥
            **kwargs: 其他配置参数

        Returns:
            LLMProvider: 支持对话的提供商实例

        Raises:
            ValueError: 提供商不支持对话功能
        """
        provider = self.create_provider(provider_name, api_key, **kwargs)

        if not provider.supports_chat():
            raise ValueError(
                f"提供商 {provider_name} 不支持对话功能。"
                f"请使用支持对话的提供商（如 dashscope, gemini）"
            )

        return provider

    def create_embedding_provider(
        self,
        provider_name: str,
        api_key: str,
        **kwargs
    ) -> LLMProvider:
        """
        创建用于嵌入的LLM提供商

        Args:
            provider_name: 提供商名称
            api_key: API密钥
            **kwargs: 其他配置参数

        Returns:
            LLMProvider: 支持嵌入的提供商实例

        Raises:
            ValueError: 提供商不支持嵌入功能
        """
        provider = self.create_provider(provider_name, api_key, **kwargs)

        if not provider.supports_embedding():
            raise ValueError(
                f"提供商 {provider_name} 不支持嵌入功能。"
                f"请使用支持嵌入的提供商（如 dashscope, gemini, openai）"
            )

        return provider

    @classmethod
    def clear_cache(cls):
        """清除缓存的提供商实例"""
        cls._providers.clear()


# 全局工厂实例
_llm_factory: Optional[LLMFactory] = None


def get_llm_factory() -> LLMFactory:
    """
    获取全局LLM工厂实例

    Returns:
        LLMFactory: 工厂实例
    """
    global _llm_factory
    if _llm_factory is None:
        _llm_factory = LLMFactory()
    return _llm_factory


def create_llm_provider(
    provider_name: str,
    api_key: str,
    **kwargs
) -> LLMProvider:
    """
    便捷函数：创建LLM提供商

    Args:
        provider_name: 提供商名称
        api_key: API密钥
        **kwargs: 其他配置参数

    Returns:
        LLMProvider: 提供商实例
    """
    factory = get_llm_factory()
    return factory.create_provider(provider_name, api_key, **kwargs)
