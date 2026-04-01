"""
数据库层单元测试
"""

import pytest
import os
import tempfile
from memory_system.storage.database import DatabaseManager, get_db_manager
from memory_system.storage.models import Memory, FileUpload, MemoryRelation
from memory_system.storage.repository import MemoryRepository, FileRepository, RelationRepository
from memory_system.storage.vector_store import VectorStore


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
def db_session(db_manager):
    """创建数据库会话"""
    with db_manager.get_session() as session:
        yield session


class TestDatabaseManager:
    """测试数据库管理器"""

    def test_init_db(self, temp_db_path):
        """测试数据库初始化"""
        manager = DatabaseManager(temp_db_path)
        manager.init_db()

        # 验证数据库文件存在
        assert os.path.exists(temp_db_path)

    def test_get_session(self, db_manager):
        """测试获取会话"""
        with db_manager.get_session() as session:
            assert session is not None

    def test_get_db_stats(self, db_manager):
        """测试获取统计信息"""
        stats = db_manager.get_db_stats()

        assert "memory_count" in stats
        assert "file_count" in stats
        assert "relation_count" in stats
        assert stats["memory_count"] == 0
        assert stats["file_count"] == 0
        assert stats["relation_count"] == 0

    def test_singleton(self, temp_db_path):
        """测试单例模式"""
        manager1 = get_db_manager(temp_db_path)
        manager2 = get_db_manager(temp_db_path)

        assert manager1 is manager2


class TestMemoryRepository:
    """测试记忆仓储"""

    def test_create_memory(self, db_manager):
        """测试创建记忆"""
        with db_manager.get_session() as session:
            repo = db_manager.get_memory_repository(session)
            memory = repo.create(
                content="测试内容",
                summary="测试摘要",
                entities=["测试实体"],
                topics=["测试主题"]
            )

            assert memory.id is not None
            assert memory.content == "测试内容"
            assert memory.summary == "测试摘要"

    def test_get_by_id(self, db_manager):
        """测试根据ID获取记忆"""
        with db_manager.get_session() as session:
            repo = db_manager.get_memory_repository(session)
            created = repo.create(content="测试内容")

            found = repo.get_by_id(created.id)
            assert found is not None
            assert found.content == "测试内容"

    def test_get_all(self, db_manager):
        """测试获取所有记忆"""
        with db_manager.get_session() as session:
            repo = db_manager.get_memory_repository(session)

            repo.create(content="内容1")
            repo.create(content="内容2")
            repo.create(content="内容3")

            memories = repo.get_all(limit=2)
            assert len(memories) == 2

    def test_search_by_text(self, db_manager):
        """测试文本搜索"""
        with db_manager.get_session() as session:
            repo = db_manager.get_memory_repository(session)

            repo.create(content="Python编程")
            repo.create(content="Java编程")
            repo.create(content="学习编程")

            results = repo.search_by_text("Python")
            assert len(results) == 1
            assert "Python" in results[0].content

    def test_search_by_entities(self, db_manager):
        """测试根据实体搜索"""
        with db_manager.get_session() as session:
            repo = db_manager.get_memory_repository(session)

            repo.create(
                content="测试1",
                entities=["Python", "编程"]
            )
            repo.create(
                content="测试2",
                entities=["Java", "编程"]
            )

            results = repo.search_by_entities(["Python"])
            assert len(results) == 1
            assert "Python" in results[0].entities

    def test_update_memory(self, db_manager):
        """测试更新记忆"""
        with db_manager.get_session() as session:
            repo = db_manager.get_memory_repository(session)
            memory = repo.create(content="原始内容")

            updated = repo.update(memory.id, summary="新摘要")
            assert updated.summary == "新摘要"

    def test_delete_memory(self, db_manager):
        """测试删除记忆"""
        with db_manager.get_session() as session:
            repo = db_manager.get_memory_repository(session)
            memory = repo.create(content="测试内容")

            # 验证存在
            assert repo.get_by_id(memory.id) is not None

            # 删除
            result = repo.delete(memory.id)
            assert result is True

            # 验证不存在
            assert repo.get_by_id(memory.id) is None

    def test_count(self, db_manager):
        """测试统计记忆数量"""
        with db_manager.get_session() as session:
            repo = db_manager.get_memory_repository(session)

            assert repo.count() == 0

            repo.create(content="内容1")
            repo.create(content="内容2")

            assert repo.count() == 2


class TestFileRepository:
    """测试文件仓储"""

    def test_create_file(self, db_manager):
        """测试创建文件记录"""
        with db_manager.get_session() as session:
            repo = db_manager.get_file_repository(session)
            file_record = repo.create(
                filename="test.txt",
                file_size=1024,
                file_type="text/plain"
            )

            assert file_record.id is not None
            assert file_record.filename == "test.txt"
            assert file_record.status == "pending"

    def test_update_status(self, db_manager):
        """测试更新状态"""
        with db_manager.get_session() as session:
            repo = db_manager.get_file_repository(session)
            file_record = repo.create(filename="test.txt")

            updated = repo.update_status(file_record.id, "completed")
            assert updated.status == "completed"

    def test_get_by_status(self, db_manager):
        """测试根据状态获取文件"""
        with db_manager.get_session() as session:
            repo = db_manager.get_file_repository(session)

            file1 = repo.create(filename="test1.txt")
            file2 = repo.create(filename="test2.txt")

            repo.update_status(file1.id, "completed")

            pending = repo.get_by_status("pending")
            assert len(pending) == 1
            assert pending[0].filename == "test2.txt"


class TestRelationRepository:
    """测试关系仓储"""

    def test_create_relation(self, db_manager):
        """测试创建关系"""
        with db_manager.get_session() as session:
            mem_repo = db_manager.get_memory_repository(session)
            rel_repo = db_manager.get_relation_repository(session)

            mem1 = mem_repo.create(content="记忆1")
            mem2 = mem_repo.create(content="记忆2")

            relation = rel_repo.create(
                memory_id=mem1.id,
                related_memory_id=mem2.id,
                relation_type="similar",
                similarity_score=0.85
            )

            assert relation.id is not None
            assert relation.memory_id == mem1.id
            assert relation.related_memory_id == mem2.id
            assert relation.similarity_score == 0.85

    def test_get_memory_relations(self, db_manager):
        """测试获取记忆的关系"""
        with db_manager.get_session() as session:
            mem_repo = db_manager.get_memory_repository(session)
            rel_repo = db_manager.get_relation_repository(session)

            mem1 = mem_repo.create(content="记忆1")
            mem2 = mem_repo.create(content="记忆2")
            mem3 = mem_repo.create(content="记忆3")

            rel_repo.create(mem1.id, mem2.id, relation_type="similar")
            rel_repo.create(mem1.id, mem3.id, relation_type="related")

            relations = rel_repo.get_memory_relations(mem1.id)
            assert len(relations) == 2


class TestVectorStore:
    """测试向量存储"""

    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]

        store = VectorStore(":memory:")

        # 正交向量相似度为0
        similarity1 = store.cosine_similarity(vec1, vec2)
        assert similarity1 == pytest.approx(0.0, abs=1e-6)

        # 相同向量相似度为1
        similarity2 = store.cosine_similarity(vec1, vec3)
        assert similarity2 == pytest.approx(1.0, abs=1e-6)

    def test_embedding_serialization(self):
        """测试向量序列化"""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # 转JSON
        json_str = VectorStore.embedding_to_json(embedding)
        assert isinstance(json_str, str)

        # 从JSON解析
        parsed = VectorStore.embedding_from_json(json_str)
        assert parsed == embedding

    def test_search_similar_memories(self, db_manager):
        """测试相似记忆搜索"""
        with db_manager.get_session() as session:
            mem_repo = db_manager.get_memory_repository(session)

            # 创建带向量的记忆
            embedding1 = [1.0, 0.0, 0.0]
            embedding2 = [0.9, 0.1, 0.0]
            embedding3 = [0.0, 1.0, 0.0]

            mem1 = mem_repo.create(content="记忆1", embedding=embedding1)
            mem2 = mem_repo.create(content="记忆2", embedding=embedding2)
            mem3 = mem_repo.create(content="记忆3", embedding=embedding3)

            # 搜索
            results = mem_repo.search_by_vector(
                query_embedding=[1.0, 0.0, 0.0],
                top_k=2,
                threshold=0.5
            )

            assert len(results) == 2
            # 第一个应该是mem1自己（相似度1.0）
            assert results[0][0].id == mem1.id
            # 第二个应该是mem2（相似度较高）
            assert results[1][0].id == mem2.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
