"""
Gemini LLM提供商实现
支持Gemini系列模型的对话API
"""

import google.genai
from google.genai import types
from typing import List, Optional, Dict
import logging

from .base import (
    LLMProvider,
    ChatMessage,
    ModelInfo,
    ChatResponse,
    EmbeddingResponse,
)
from memory_system.utils.logger import get_llm_logger


class GeminiProvider(LLMProvider):
    """
    Gemini提供商实现
    支持Gemini系列模型的对话功能
    """

    def __init__(self, api_key: str, **kwargs):
        """
        初始化Gemini提供商

        Args:
            api_key: Google API密钥
            **kwargs: 其他配置参数
                - model: 默认模型名称 (默认: gemini-2.0-flash)
                - temperature: 默认温度 (默认: 0.7)
        """
        super().__init__(api_key, **kwargs)
        self.logger = get_llm_logger()

        # 配置客户端
        self.client = google.genai.Client(api_key=api_key)

        # 默认配置
        self.default_model = kwargs.get("model", "gemini-2.0-flash")
        self.default_temperature = kwargs.get("temperature", 0.7)

        self.logger.info(f"GeminiProvider初始化成功，模型: {self.default_model}")

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
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            ChatResponse: 对话响应
        """
        model_name = model or self.default_model

        try:
            # 转换消息格式
            contents = []
            system_instruction = None

            for msg in messages:
                if msg.role == "system":
                    system_instruction = msg.content
                else:
                    role = "user" if msg.role == "user" else "model"
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part(text=msg.content)]
                        )
                    )

            # 配置生成参数
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )

            # 发送请求
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            except Exception as e:
                if "models/" not in model_name:
                    self.logger.warning(f"尝试带前缀的模型名称: models/{model_name}")
                    response = self.client.models.generate_content(
                        model=f"models/{model_name}",
                        contents=contents,
                        config=config
                    )
                    model_name = f"models/{model_name}"
                else:
                    raise e

            self.logger.info(f"Gemini对话成功，模型: {model_name}")

            # 提取token统计信息
            usage: Dict[str, int] = {}
            if response.usage_metadata:
                if response.usage_metadata.prompt_token_count is not None:
                    usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
                if response.usage_metadata.candidates_token_count is not None:
                    usage["completion_tokens"] = response.usage_metadata.candidates_token_count
                if response.usage_metadata.total_token_count is not None:
                    usage["total_tokens"] = response.usage_metadata.total_token_count

            return ChatResponse(
                content=response.text or "",
                model=model_name,
                usage=usage,
                raw_response=response,
            )

        except Exception as e:
            self.logger.error(f"Gemini对话失败: {str(e)}")
            raise

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Gemini提供Embedding API
        """
        model_name = model or "gemini-embedding-001"
        try:
            response = self.client.models.embed_content(
                model=model_name,
                contents=texts,
                **kwargs
            )
            
            embeddings: List[List[float]] = []
            if response.embeddings:
                for item in response.embeddings:
                    if item.values:
                        embeddings.append(item.values)
            
            dimensions = len(embeddings[0]) if embeddings else 0
            
            # 使用 metadata 统计 token
            usage: Dict[str, int] = {}
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=model_name,
                dimensions=dimensions,
                usage=usage
            )
        except Exception as e:
            self.logger.error(f"Gemini Embedding失败: {str(e)}")
            raise

    def get_model_info(self) -> ModelInfo:
        """
        获取模型信息

        Returns:
            ModelInfo: 模型信息
        """
        return ModelInfo(
            provider="Gemini",
            model_name=self.default_model,
            supports_chat=True,
            supports_embedding=True,
            max_tokens=8192,
            description="Google Gemini模型，支持对话和Embedding功能"
        )

    def supports_chat(self) -> bool:
        """支持对话"""
        return True

    def supports_embedding(self) -> bool:
        """支持embedding"""
        return True
