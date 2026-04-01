"""
LLM模块单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from memory_system.llm import (
    LLMProvider,
    ChatMessage,
    ModelInfo,
    ChatResponse,
    EmbeddingResponse,
    GeminiProvider,
    DashScopeProvider,
    OpenAIProvider,
    LLMFactory,
    create_llm_provider,
)


class TestChatMessage:
    """测试ChatMessage数据类"""

    def test_create_chat_message(self):
        """测试创建聊天消息"""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestGeminiProvider:
    """测试Gemini提供商"""

    def test_init(self):
        """测试初始化"""
        provider = GeminiProvider(api_key="test_key")
        assert provider.api_key == "test_key"

    @patch("memory_system.llm.gemini_provider.google.genai")
    def test_supports_chat(self, mock_genai):
        """测试是否支持对话"""
        provider = GeminiProvider(api_key="test_key")
        assert provider.supports_chat() is True

    @patch("memory_system.llm.gemini_provider.google.genai")
    def test_supports_embedding(self, mock_genai):
        """测试是否支持embedding"""
        provider = GeminiProvider(api_key="test_key")
        assert provider.supports_embedding() is True

    @patch("memory_system.llm.gemini_provider.google.genai")
    def test_get_model_info(self, mock_genai):
        """测试获取模型信息"""
        provider = GeminiProvider(api_key="test_key")
        info = provider.get_model_info()
        assert info.provider == "Gemini"
        assert info.supports_chat is True
        assert info.supports_embedding is True

    @patch("memory_system.llm.gemini_provider.google.genai")
    def test_embed_with_mock(self, mock_genai):
        """测试Gemini embedding（使用Mock）"""
        # Mock 客户端和嵌入响应
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        
        mock_client.models.embed_content.return_value = mock_response
        
        provider = GeminiProvider(api_key="test_key")
        response = provider.embed(["test text"])
        
        assert len(response.embeddings) == 1
        assert response.dimensions == 3
        assert response.embeddings[0] == [0.1, 0.2, 0.3]


class TestDashScopeProvider:
    """测试DashScope提供商"""

    def test_init(self):
        """测试初始化"""
        provider = DashScopeProvider(api_key="test_key")
        assert provider.api_key == "test_key"

    def test_supports_chat(self):
        """测试是否支持对话"""
        provider = DashScopeProvider(api_key="test_key")
        assert provider.supports_chat() is True

    def test_supports_embedding(self):
        """测试是否支持embedding"""
        provider = DashScopeProvider(api_key="test_key")
        assert provider.supports_embedding() is True

    def test_get_model_info(self):
        """测试获取模型信息"""
        provider = DashScopeProvider(api_key="test_key")
        info = provider.get_model_info()
        assert info.provider == "DashScope"
        assert info.supports_chat is True
        assert info.supports_embedding is True

    @patch("memory_system.llm.dashscope_provider.Generation")
    def test_chat_with_mock(self, mock_generation):
        """测试对话（使用Mock）"""
        # Mock响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.output.choices = [
            Mock(message=Mock(content="Test response"))
        ]
        mock_response.usage = Mock(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15
        )
        mock_generation.call.return_value = mock_response

        provider = DashScopeProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]

        response = provider.chat(messages)

        assert response.content == "Test response"
        assert response.usage["total_tokens"] == 15

    @patch("memory_system.llm.dashscope_provider.TextEmbedding")
    def test_embed_with_mock(self, mock_embedding):
        """测试Embedding（使用Mock）"""
        # Mock响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.output.embeddings = [
            Mock(embedding=[0.1, 0.2, 0.3])
        ]
        mock_response.usage = Mock(total_tokens=10)
        mock_embedding.call.return_value = mock_response

        provider = DashScopeProvider(api_key="test_key")
        response = provider.embed(["test text"])

        assert len(response.embeddings) == 1
        assert response.dimensions == 3
        assert response.embeddings[0] == [0.1, 0.2, 0.3]


class TestOpenAIProvider:
    """测试OpenAI提供商"""

    def test_init(self):
        """测试初始化"""
        provider = OpenAIProvider(api_key="test_key", base_url="https://api.test.com/v1")
        assert provider.api_key == "test_key"
        assert provider.base_url == "https://api.test.com/v1"

    def test_supports_chat(self):
        """测试是否支持对话"""
        provider = OpenAIProvider(api_key="test_key")
        assert provider.supports_chat() is True

    def test_supports_embedding(self):
        """测试是否支持embedding"""
        provider = OpenAIProvider(api_key="test_key")
        assert provider.supports_embedding() is True

    def test_get_model_info(self):
        """测试获取模型信息"""
        provider = OpenAIProvider(api_key="test_key", base_url="https://api.test.com/v1")
        info = provider.get_model_info()
        assert info.provider == "OpenAI"
        assert "https://api.test.com/v1" in info.description

    @patch("memory_system.llm.openai_provider.OpenAI")
    def test_chat_with_mock(self, mock_openai):
        """测试对话（使用Mock）"""
        # Mock响应
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Test response"))
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]

        response = provider.chat(messages)

        assert response.content == "Test response"
        assert response.usage["total_tokens"] == 15

    @patch("memory_system.llm.openai_provider.OpenAI")
    def test_embed_with_mock(self, mock_openai):
        """测试Embedding（使用Mock）"""
        # Mock响应
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]
        
        mock_response = MagicMock()
        mock_response.data = [mock_item]
        mock_response.usage = MagicMock(total_tokens=10)
        
        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")
        response = provider.embed(["test text"])

        assert len(response.embeddings) == 1
        assert response.dimensions == 3
        assert response.embeddings[0] == [0.1, 0.2, 0.3]


class TestLLMFactory:
    """测试LLM工厂"""

    def test_singleton(self):
        """测试单例模式"""
        factory1 = LLMFactory()
        factory2 = LLMFactory()
        assert factory1 is factory2

    def test_create_gemini_provider(self):
        """测试创建Gemini提供商"""
        factory = LLMFactory()
        provider = factory.create_provider("gemini", "test_key")
        assert isinstance(provider, GeminiProvider)

    def test_create_dashscope_provider(self):
        """测试创建DashScope提供商"""
        factory = LLMFactory()
        provider = factory.create_provider("dashscope", "test_key")
        assert isinstance(provider, DashScopeProvider)

    def test_create_invalid_provider(self):
        """测试创建不支持的提供商"""
        factory = LLMFactory()
        with pytest.raises(ValueError, match="不支持的LLM提供商"):
            factory.create_provider("invalid", "test_key")

    def test_get_or_create_provider_cache(self):
        """测试缓存机制"""
        factory = LLMFactory()
        provider1 = factory.get_or_create_provider("dashscope", "test_key")
        provider2 = factory.get_or_create_provider("dashscope", "test_key")

        # 同一个API密钥应该返回同一个实例
        assert provider1 is provider2

    def test_create_chat_provider(self):
        """测试创建对话提供商"""
        factory = LLMFactory()
        provider = factory.create_chat_provider("dashscope", "test_key")
        assert isinstance(provider, DashScopeProvider)

    def test_create_embedding_provider(self):
        """测试创建embedding提供商"""
        factory = LLMFactory()
        provider = factory.create_embedding_provider("dashscope", "test_key")
        assert isinstance(provider, DashScopeProvider)

    def test_create_embedding_provider_with_gemini(self):
        """测试Gemini支持embedding"""
        factory = LLMFactory()
        provider = factory.create_embedding_provider("gemini", "test_key")
        assert isinstance(provider, GeminiProvider)
        assert provider.supports_embedding() is True


class TestCreateLLMProvider:
    """测试便捷函数"""

    def test_create_llm_provider(self):
        """测试创建提供商便捷函数"""
        provider = create_llm_provider("dashscope", "test_key")
        assert isinstance(provider, DashScopeProvider)


def test_get_llm_factory():
    """测试获取工厂便捷函数"""
    from memory_system.llm.factory import get_llm_factory
    factory = get_llm_factory()
    assert isinstance(factory, LLMFactory)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
