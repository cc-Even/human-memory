"""
IngestAgent单元测试
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

from memory_system.agents.ingest_agent import IngestAgent
from memory_system.llm import ChatResponse, EmbeddingResponse
from memory_system.storage.database import DatabaseManager
from memory_system.storage.models import Memory


@pytest.fixture
def temp_db_path():
    """创建临时数据库路径"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # 清理
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def db_manager(temp_db_path):
    """创建数据库管理器"""
    manager = DatabaseManager(temp_db_path)
    manager.init_db()
    return manager


@pytest.fixture
def mock_llm_provider():
    """创建模拟的LLM提供商"""
    provider = Mock()

    # 模拟chat方法
    def mock_chat(messages, temperature, max_tokens):
        prompt = messages[-1].content
        if "总结" in prompt:
            return ChatResponse(content="这是测试摘要", model="mock-model")
        elif "实体" in prompt:
            return ChatResponse(content='["Python", "编程", "测试"]', model="mock-model")
        elif "主题" in prompt:
            return ChatResponse(content='["技术", "学习"]', model="mock-model")
        else:
            return ChatResponse(content="默认响应")

    provider.chat = mock_chat

    # 模拟embed方法
    def mock_embed(texts):
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts], dimensions=5,
            model="test-model",
            usage={"total_tokens": 100}
        )

    provider.embed = mock_embed
    provider.supports_embedding = Mock(return_value=True)

    return provider


@pytest.fixture
def ingest_agent(db_manager, mock_llm_provider):
    """创建IngestAgent实例"""
    return IngestAgent(
        llm_provider=mock_llm_provider,
        db_manager=db_manager
    )


class TestIngestAgent:
    """测试IngestAgent"""

    def test_init(self, db_manager, mock_llm_provider):
        """测试初始化"""
        agent = IngestAgent(
            llm_provider=mock_llm_provider,
            db_manager=db_manager
        )

        assert agent.name == "IngestAgent"
        assert agent.db_manager == db_manager
        assert agent.llm_provider == mock_llm_provider

    def test_summarize_content(self, ingest_agent):
        """测试内容总结"""
        content = "这是一段很长的测试内容，需要被总结成简洁的摘要。"

        summary = ingest_agent.summarize_content(content)

        assert summary == "这是测试摘要"

    def test_extract_entities(self, ingest_agent):
        """测试实体提取"""
        content = "Python是一门编程语言，我喜欢编程测试。"

        entities = ingest_agent.extract_entities(content)

        assert isinstance(entities, list)
        assert len(entities) == 3
        assert "Python" in entities
        assert "编程" in entities

    def test_tag_topics(self, ingest_agent):
        """测试主题打标"""
        content = "这是一篇关于技术的学习文章。"

        topics = ingest_agent.tag_topics(content)

        assert isinstance(topics, list)
        assert "技术" in topics
        assert "学习" in topics

    def test_create_embedding(self, ingest_agent):
        """测试创建向量"""
        content = "测试向量化"

        embedding = ingest_agent.create_embedding(content)

        assert isinstance(embedding, list)
        assert len(embedding) == 5
        assert embedding[0] == 0.1

    def test_ingest_text(self, ingest_agent):
        """测试摄入文本"""
        text = "这是用户输入的测试文本内容。"

        memory = ingest_agent.ingest_text(text)

        assert isinstance(memory, Memory)
        assert memory.id is not None
        assert memory.content == text
        assert memory.summary == "这是测试摘要"
        assert "Python" in memory.entities
        assert "技术" in memory.topics
        assert memory.source_type == "user_input"
        assert len(memory.embedding) == 5 or len(memory.embedding) > 5

    def test_process(self, ingest_agent):
        """测试处理内容"""
        content = "测试处理流程"

        memory = ingest_agent.process(
            content=content,
            source_type="test",
            source_id=123
        )

        assert memory.content == content
        assert memory.source_type == "test"
        assert memory.source_id == 123

    def test_ingest_file(self, ingest_agent, temp_db_path):
        """测试摄入文件"""
        # 创建测试文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("这是文件内容")
            test_file = f.name

        try:
            memory = ingest_agent.ingest_file(test_file)

            assert isinstance(memory, Memory)
            assert memory.content == "这是文件内容"
            assert memory.source_type == "file_upload"

            # 验证文件记录
            with ingest_agent.db_manager.get_session() as session:
                file_repo = ingest_agent.db_manager.get_file_repository(session)
                files = file_repo.get_by_status("completed")
                assert len(files) == 1
                assert files[0].filename == os.path.basename(test_file)

        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_ingest_json_file(self, ingest_agent):
        """测试摄入JSON文件"""
        # 创建测试JSON文件
        test_data = {"key": "value", "number": 123}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)
            test_file = f.name

        try:
            memory = ingest_agent.ingest_file(test_file)

            assert isinstance(memory, Memory)
            assert "key" in memory.content
            assert "value" in memory.content

        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_ingest_file_not_found(self, ingest_agent):
        """测试摄入不存在的文件"""
        with pytest.raises(FileNotFoundError):
            ingest_agent.ingest_file("/nonexistent/file.txt")

    def test_extract_entities_json_error(self, ingest_agent):
        """测试实体提取JSON解析失败"""
        # 模拟返回无效JSON
        ingest_agent.llm_provider.chat = Mock(
            return_value=ChatResponse(content="不是有效的JSON格式", model="mock-model")
        )

        entities = ingest_agent.extract_entities("测试内容")

        assert entities == []

    def test_tag_topics_json_error(self, ingest_agent):
        """测试主题打标JSON解析失败"""
        # 模拟返回无效JSON
        ingest_agent.llm_provider.chat = Mock(
            return_value=ChatResponse(content="不是有效的JSON格式", model="mock-model")
        )

        topics = ingest_agent.tag_topics("测试内容")

        assert topics == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
