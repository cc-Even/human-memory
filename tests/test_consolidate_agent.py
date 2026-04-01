"""
ConsolidateAgent单元测试
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import tempfile
import os

from memory_system.agents.consolidate_agent import ConsolidateAgent
from memory_system.llm import ChatResponse
from memory_system.storage.database import DatabaseManager
from memory_system.storage.models import Memory, MemoryRelation


class TestConsolidateAgent(unittest.TestCase):
    """测试ConsolidateAgent"""

    def setUp(self):
        """测试前准备"""
        # 创建临时数据库
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)

        # 创建数据库管理器
        self.db_manager = DatabaseManager(self.temp_db_path)
        self.db_manager.init_db()

        # 创建模拟的LLM提供商
        self.mock_llm_provider = Mock()

        def mock_chat(messages, temperature, max_tokens):
            prompt = messages[-1].content
            if "关键词" in prompt:
                return ChatResponse(content='["Python", "编程"]', model="mock-model")
            elif "判断" in prompt:
                return ChatResponse(content='{"related": true, "score": 0.85, "reason": "同一主题"}', model="mock-model")
            elif "合并" in prompt:
                return ChatResponse(content='{"content": "合并后的完整内容", "summary": "新摘要", "entities": ["Python"], "topics": ["编程"]}', model="mock-model")
            else:
                return ChatResponse(content="默认响应")

        self.mock_llm_provider.chat = mock_chat
        self.mock_llm_provider.supports_embedding = Mock(return_value=False)

        # 创建ConsolidateAgent实例
        self.agent = ConsolidateAgent(
            llm_provider=self.mock_llm_provider,
            db_manager=self.db_manager
        )

        # 创建示例记忆
        self.sample_memories = self._create_sample_memories()

    def _create_sample_memories(self):
        """创建示例记忆数据"""
        with self.db_manager.get_session() as session:
            mem_repo = self.db_manager.get_memory_repository(session)

            mem1 = mem_repo.create(
                content="Python是一门编程语言",
                summary="Python编程",
                entities=["Python", "编程"],
                topics=["技术", "学习"]
            )

            mem2 = mem_repo.create(
                content="我正在学习Python编程",
                summary="学习Python",
                entities=["Python", "学习"],
                topics=["学习", "编程"]
            )

            mem3 = mem_repo.create(
                content="Java也是一门编程语言",
                summary="Java编程",
                entities=["Java", "编程"],
                topics=["技术", "学习"]
            )

            return [mem1, mem2, mem3]

    def tearDown(self):
        """测试后清理"""
        # 删除临时数据库
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.agent.name, "ConsolidateAgent")
        self.assertEqual(self.agent.db_manager, self.db_manager)

    def test_generate_keywords(self):
        """测试生成关键词"""
        keywords = self.agent.generate_keywords("Python编程语言测试")

        self.assertIsInstance(keywords, list)
        self.assertIn("Python", keywords)
        self.assertIn("编程", keywords)

    def test_judge_related(self):
        """测试判断关联"""
        mem1, mem2, mem3 = self.sample_memories

        result = self.agent.judge_related(mem1, mem2)

        self.assertTrue(result["related"])
        self.assertEqual(result["score"], 0.85)
        self.assertEqual(result["reason"], "同一主题")

    def test_find_related_memories(self):
        """测试查找相关记忆"""
        mem1, mem2, mem3 = self.sample_memories

        related = self.agent.find_related_memories(
            keywords=["Python", "编程"],
            exclude_id=mem1.id,
            limit=2
        )

        self.assertEqual(len(related), 2)
        self.assertNotIn(mem1, related)
        self.assertTrue(any(r.id == mem2.id for r in related))

    def test_process(self):
        """测试处理单个记忆"""
        mem1, mem2, mem3 = self.sample_memories

        result = self.agent.process(mem1.id)

        self.assertTrue(result["success"])
        self.assertEqual(result["memory_id"], mem1.id)
        self.assertGreaterEqual(result["relations_created"], 0)

        # 验证创建了关系
        with self.db_manager.get_session() as session:
            rel_repo = self.db_manager.get_relation_repository(session)
            relations = rel_repo.get_memory_relations(mem1.id)
            self.assertGreater(len(relations), 0)

    def test_consolidate(self):
        """测试批量整合"""
        result = self.agent.consolidate(time_window_hours=24)

        self.assertGreaterEqual(result["total"], 0)
        self.assertGreaterEqual(result["processed"], 0)
        self.assertIn("relations_created", result)

    def test_find_patterns(self):
        """测试发现规律"""
        patterns = self.agent.find_patterns(limit=100)

        self.assertIsInstance(patterns, list)
        # 应该发现一些规律（实体或主题）
        self.assertGreater(len(patterns), 0)

        # 检查规律结构
        for pattern in patterns:
            self.assertIn("type", pattern)
            self.assertIn("name", pattern)
            self.assertIn("frequency", pattern)

    def test_merge_memories(self):
        """测试合并记忆"""
        mem1, mem2, mem3 = self.sample_memories

        # 合并mem1和mem2
        new_memory = self.agent.merge_memories([mem1.id, mem2.id])

        self.assertIsNotNone(new_memory)
        self.assertEqual(new_memory.source_type, "consolidation")
        self.assertIn("合并后的完整内容", new_memory.content)
        self.assertEqual(new_memory.summary, "新摘要")

        # 验证原记忆被标记为已合并
        with self.db_manager.get_session() as session:
            mem_repo = self.db_manager.get_memory_repository(session)
            updated_mem1 = mem_repo.get_by_id(mem1.id)
            updated_mem2 = mem_repo.get_by_id(mem2.id)

            self.assertEqual(updated_mem1.is_merged, 1)
            self.assertEqual(updated_mem2.is_merged, 1)
            self.assertEqual(updated_mem1.merged_into, new_memory.id)
            self.assertEqual(updated_mem2.merged_into, new_memory.id)

    def test_merge_memories_insufficient(self):
        """测试合并记忆数量不足"""
        mem1, _, _ = self.sample_memories

        result = self.agent.merge_memories([mem1.id])

        self.assertIsNone(result)

    def test_judge_related_json_error(self):
        """测试关联判断JSON解析失败"""
        # 模拟返回无效JSON
        self.agent.llm_provider.chat = Mock(
            return_value=ChatResponse(content="不是有效的JSON格式", model="mock-model")
        )

        mem1, mem2, mem3 = self.sample_memories
        result = self.agent.judge_related(mem1, mem2)

        self.assertFalse(result["related"])
        self.assertEqual(result["score"], 0.0)

    def test_find_related_memories_no_keywords(self):
        """测试无关键词时查找相关记忆"""
        related = self.agent.find_related_memories(keywords=[])

        self.assertEqual(related, [])

    def test_config(self):
        """测试配置"""
        self.assertEqual(self.agent.similarity_threshold, 0.75)
        self.assertEqual(self.agent.top_k, 5)

        # 修改配置
        self.agent.set_config("similarity_threshold", 0.8)
        self.assertEqual(self.agent.get_config("similarity_threshold"), 0.8)

        # 获取不存在的配置
        self.assertEqual(self.agent.get_config("nonexistent", "default"), "default")


if __name__ == "__main__":
    unittest.main()
