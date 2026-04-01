"""
LLM提供商抽象基类
定义统一的接口规范
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """聊天消息数据类"""
    role: str  # system, user, assistant
    content: str


@dataclass
class ModelInfo:
    """模型信息数据类"""
    provider: str
    model_name: str
    supports_chat: bool
    supports_embedding: bool
    max_tokens: Optional[int] = None
    description: str = ""


@dataclass
class ChatResponse:
    """聊天响应数据类"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


@dataclass
class EmbeddingResponse:
    """嵌入响应数据类"""
    embeddings: List[List[float]]  # 每个文本的向量表示
    model: str
    dimensions: int
    usage: Optional[Dict[str, int]] = None


class LLMProvider(ABC):
    """
    LLM提供商抽象基类

    所有具体的LLM提供商（Gemini、DashScope等）都需要实现这个接口
    """

    def __init__(self, api_key: str, **kwargs):
        """
        初始化LLM提供商

        Args:
            api_key: API密钥
            **kwargs: 其他配置参数
        """
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """
        发起对话请求

        Args:
            messages: 消息列表，包含role和content
            model: 使用的模型名称（可选，使用默认模型）
            temperature: 温度参数，控制随机性 (0-1)
            max_tokens: 最大生成token数
            **kwargs: 其他模型特定参数

        Returns:
            ChatResponse: 包含响应内容、模型信息和使用情况

        Raises:
            Exception: API调用失败时抛出异常
        """
        pass

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        将文本转换为向量表示

        Args:
            texts: 需要向量化的文本列表
            model: 使用的embedding模型（可选）
            **kwargs: 其他模型特定参数

        Returns:
            EmbeddingResponse: 包含向量列表、模型信息和维度

        Raises:
            Exception: API调用失败时抛出异常
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        获取模型信息

        Returns:
            ModelInfo: 包含模型详细信息的对象
        """
        pass

    @abstractmethod
    def supports_chat(self) -> bool:
        """是否支持对话功能"""
        pass

    @abstractmethod
    def supports_embedding(self) -> bool:
        """是否支持嵌入功能"""
        pass

    def chat_with_history(
        self,
        user_message: str,
        history: Optional[List[ChatMessage]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """
        基于历史记录的对话（便捷方法）

        Args:
            user_message: 用户消息
            history: 历史消息列表
            system_prompt: 系统提示词
            **kwargs: 其他参数

        Returns:
            ChatResponse: 对话响应
        """
        messages = []

        # 添加系统提示词
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))

        # 添加历史消息
        if history:
            messages.extend(history)

        # 添加当前用户消息
        messages.append(ChatMessage(role="user", content=user_message))

        return self.chat(messages, **kwargs)
