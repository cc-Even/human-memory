"""
阿里云DashScope LLM提供商实现
支持Qwen系列模型的对话API和Embedding API
"""

import dashscope
from dashscope import Generation, TextEmbedding
from typing import List, Optional, Dict, Any
import logging
import json

from .base import (
    LLMProvider,
    ChatMessage,
    ModelInfo,
    ChatResponse,
    EmbeddingResponse,
)
from memory_system.utils.logger import get_llm_logger


class DashScopeProvider(LLMProvider):
    """
    DashScope提供商实现
    支持Qwen系列模型的对话和Embedding功能
    """

    def __init__(self, api_key: str, **kwargs):
        """
        初始化DashScope提供商

        Args:
            api_key: 阿里云API密钥
            **kwargs: 其他配置参数
                - chat_model: 默认对话模型 (默认: qwen-plus)
                - embedding_model: 默认embedding模型 (默认: text-embedding-v2)
                - temperature: 默认温度 (默认: 0.7)
        """
        super().__init__(api_key, **kwargs)
        self.logger = get_llm_logger()

        # 配置API密钥
        dashscope.api_key = api_key

        # 默认配置
        self.default_chat_model = kwargs.get("chat_model", "qwen-plus")
        self.default_embedding_model = kwargs.get("embedding_model", "text-embedding-v2")
        self.default_temperature = kwargs.get("temperature", 0.7)

        self.logger.info(
            f"DashScopeProvider初始化成功，对话模型: {self.default_chat_model}, "
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
        
        # 显式设置 API Key
        import dashscope
        dashscope.api_key = self.api_key

        try:
            # 转换消息格式
            dashscope_messages = []
            for msg in messages:
                dashscope_messages.append(
                    {"role": msg.role, "content": msg.content}
                )

            # 准备参数
            gen_params = {
                "model": model_name,
                "messages": dashscope_messages,
                "result_format": "message",
                "temperature": temperature,
            }

            if max_tokens:
                gen_params["max_tokens"] = max_tokens

            # 添加额外参数
            gen_params.update(kwargs)

            self.logger.info(f"调用DashScope对话API，模型: {model_name}")

            # 调用API
            response = Generation.call(api_key=self.api_key, **gen_params)

            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"DashScope API调用失败: {response.code} - {response.message}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

            # 提取响应内容
            content = response.output.choices[0].message.content
            usage = response.usage

            self.logger.info(f"DashScope对话成功，消耗tokens: {usage.total_tokens}")

            return ChatResponse(
                content=content,
                model=model_name,
                usage={
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                },
                raw_response=response,
            )

        except Exception as e:
            self.logger.error(f"DashScope对话失败: {str(e)}")
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

        # 显式设置 API Key
        import dashscope
        dashscope.api_key = self.api_key

        try:
            self.logger.info(
                f"调用DashScope Embedding API，模型: {model_name}，文本数量: {len(texts)}"
            )

            # 准备参数
            embed_params = {
                "model": model_name,
                "input": texts,
                "parameters": kwargs.get("parameters", {})
            }

            # 调用API
            response = TextEmbedding.call(api_key=self.api_key, **embed_params)

            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"DashScope Embedding API调用失败: {response.code} - {response.message}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

            # 提取向量
            embeddings = []
            if hasattr(response.output, 'embeddings'):
                for item in response.output.embeddings:
                    embeddings.append(item.embedding)
            elif isinstance(response.output, dict) and 'embeddings' in response.output:
                for item in response.output['embeddings']:
                    embeddings.append(item['embedding'])
            else:
                self.logger.error(f"无法从DashScope响应中提取向量: {response.output}")
                raise Exception("Failed to extract embeddings from DashScope response")

            dimensions = len(embeddings[0]) if embeddings else 0
            
            # 安全地提取 usage
            total_tokens = 0
            if hasattr(response, 'usage'):
                usage = response.usage
                if hasattr(usage, 'total_tokens'):
                    total_tokens = usage.total_tokens
                elif isinstance(usage, dict) and 'total_tokens' in usage:
                    total_tokens = usage['total_tokens']

            self.logger.info(
                f"DashScope Embedding成功，向量维度: {dimensions}，消耗tokens: {total_tokens}"
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
            self.logger.error(f"DashScope Embedding失败: {str(e)}")
            raise

    def get_model_info(self) -> ModelInfo:
        """
        获取模型信息

        Returns:
            ModelInfo: 模型信息
        """
        return ModelInfo(
            provider="DashScope",
            model_name=f"{self.default_chat_model}/{self.default_embedding_model}",
            supports_chat=True,
            supports_embedding=True,
            max_tokens=8000,
            description="阿里云DashScope模型，支持对话和Embedding功能"
        )

    def supports_chat(self) -> bool:
        """支持对话"""
        return True

    def supports_embedding(self) -> bool:
        """支持embedding"""
        return True

    def chat_stream(self, messages: List[ChatMessage], **kwargs) -> Any:
        """
        流式对话（可选功能）

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Yields:
            str: 流式返回的文本片段
        """
        model_name = kwargs.get("model", self.default_chat_model)

        try:
            # 转换消息格式
            dashscope_messages = []
            for msg in messages:
                dashscope_messages.append(
                    {"role": msg.role, "content": msg.content}
                )

            # 准备参数
            gen_params = {
                "model": model_name,
                "messages": dashscope_messages,
                "result_format": "message",
                "stream": True,
            }

            gen_params.update(kwargs)

            self.logger.info(f"调用DashScope流式对话API，模型: {model_name}")

            # 调用流式API
            response = Generation.call(**gen_params)

            for chunk in response:
                if chunk.status_code == 200:
                    content = chunk.output.choices[0].message.content
                    if content:
                        yield content
                else:
                    error_msg = f"流式对话失败: {chunk.code} - {chunk.message}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)

        except Exception as e:
            self.logger.error(f"DashScope流式对话失败: {str(e)}")
            raise
