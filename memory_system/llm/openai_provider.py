"""
OpenAI LLM提供商实现
支持OpenAI系列模型的对话API和Embedding API
"""

from openai import OpenAI
from typing import List, Optional, Dict, Any
import logging

from .base import (
    LLMProvider,
    ChatMessage,
    ModelInfo,
    ChatResponse,
    EmbeddingResponse,
)
from memory_system.utils.logger import get_llm_logger


class OpenAIProvider(LLMProvider):
    """
    OpenAI提供商实现
    支持OpenAI系列模型的对话和Embedding功能
    """

    def __init__(self, api_key: str, **kwargs):
        """
        初始化OpenAI提供商

        Args:
            api_key: OpenAI API密钥
            **kwargs: 其他配置参数
                - base_url: API基础URL (可选)
                - chat_model: 默认对话模型 (默认: gpt-3.5-turbo)
                - embedding_model: 默认embedding模型 (默认: text-embedding-3-small)
                - temperature: 默认温度 (默认: 0.7)
        """
        super().__init__(api_key, **kwargs)
        self.logger = get_llm_logger()

        # 配置API基础URL
        self.base_url = kwargs.get("base_url")

        # 初始化客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url
        )

        # 默认配置
        self.default_chat_model = kwargs.get("chat_model", "gpt-3.5-turbo")
        self.default_embedding_model = kwargs.get("embedding_model", "text-embedding-3-small")
        self.default_temperature = kwargs.get("temperature", 0.7)

        self.logger.info(
            f"OpenAIProvider初始化成功，基础URL: {self.base_url or '默认'}, "
            f"对话模型: {self.default_chat_model}, "
            f"Embedding模型: {self.default_embedding_model}"
        )

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
        """
        model_name = model or self.default_chat_model

        try:
            # 转换消息格式
            openai_messages = []
            for msg in messages:
                openai_messages.append(
                    {"role": msg.role, "content": msg.content}
                )

            # 准备参数
            gen_params = {
                "model": model_name,
                "messages": openai_messages,
                "temperature": temperature,
            }

            if max_tokens:
                gen_params["max_tokens"] = max_tokens

            # 添加额外参数
            gen_params.update(kwargs)

            self.logger.info(f"调用OpenAI对话API，模型: {model_name}")

            # 调用API
            response = self.client.chat.completions.create(**gen_params)  # type: ignore

            # 提取响应内容
            content = response.choices[0].message.content
            usage = response.usage

            if usage:
                self.logger.info(f"OpenAI对话成功，消耗tokens: {usage.total_tokens}")

                return ChatResponse(
                    content=content or "",
                    model=model_name,
                    usage={
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    raw_response=response,
                )
            else:
                self.logger.info("OpenAI对话成功，但未返回消耗信息")
                return ChatResponse(
                    content=content or "",
                    model=model_name,
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    raw_response=response,
                )

        except Exception as e:
            self.logger.error(f"OpenAI对话失败: {str(e)}")
            raise

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        将文本转换为向量表示
        """
        model_name = model or self.default_embedding_model

        try:
            self.logger.info(
                f"调用OpenAI Embedding API，模型: {model_name}，文本数量: {len(texts)}"
            )

            # 调用API
            response = self.client.embeddings.create(
                input=texts,
                model=model_name,
                **kwargs
            )

            # 提取向量
            embeddings = [item.embedding for item in response.data]
            dimensions = len(embeddings[0]) if embeddings else 0
            
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0

            self.logger.info(
                f"OpenAI Embedding成功，向量维度: {dimensions}，消耗tokens: {total_tokens}"
            )

            return EmbeddingResponse(
                embeddings=embeddings,
                model=model_name,
                dimensions=dimensions,
                usage={
                    "total_tokens": total_tokens,
                },
            )

        except Exception as e:
            self.logger.error(f"OpenAI Embedding失败: {str(e)}")
            raise

    def get_model_info(self) -> ModelInfo:
        """
        获取模型信息

        Returns:
            ModelInfo: 模型信息
        """
        return ModelInfo(
            provider="OpenAI",
            model_name=f"{self.default_chat_model}/{self.default_embedding_model}",
            supports_chat=True,
            supports_embedding=True,
            max_tokens=None,
            description=f"OpenAI模型，基础URL: {self.base_url or '默认'}"
        )

    def supports_chat(self) -> bool:
        """支持对话"""
        return True

    def supports_embedding(self) -> bool:
        """支持embedding"""
        return True
