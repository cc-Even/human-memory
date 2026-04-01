"""
QueryAgent单元测试
测试查询智能体的各项功能
"""

import unittest
import sys
import os

from unittest.mock import Mock, MagicMock, patch

# Mock所有外部依赖模块（避免依赖问题）
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()
sys.modules['dashscope'] = MagicMock()

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory_system.agents.query_agent import QueryAgent
from memory_system.storage.models import Memory


class TestQueryAgent(unittest.TestCase):
    """QueryAgent测试类"""

    def setUp(self):
        """测试前置准备"""
        # 创建mock对象
        self.mock_llm = Mock()
        self.mock_llm.supports_embedding.return_value = False

        self.mock_db_manager = Mock()
        self.mock_session = MagicMock()
        self.mock_db_manager.get_session.return_value = MagicMock(__enter__=MagicMock(return_value=self.mock_session), __exit__=MagicMock(return_value=None))

        # 创建mock repositories
        self.mock_mem_repo = Mock()
        self.mock_rel_repo = Mock()
        self.mock_db_manager.get_memory_repository.return_value = self.mock_mem_repo
        self.mock_db_manager.get_relation_repository.return_value = self.mock_rel_repo

        # 配置
        self.config = {
            "top_k": 3,
            "relevance_threshold": 0.3,
            "temperature": 0.7
        }

        # 创建QueryAgent实例
        self.query_agent = QueryAgent(
            llm_provider=self.mock_llm,
            db_manager=self.mock_db_manager,
            config=self.config
        )

        # 创建测试记忆数据
        self.test_memories = [
            Memory(
                id=1,
                content="我喜欢吃苹果和香蕉",
                summary="水果喜好",
                source_type="explicit",
                embedding=None
            ),
            Memory(
                id=2,
                content="昨天去超市买了一些橙子和葡萄",
                summary="购物记录",
                source_type="explicit",
                embedding=None
            ),
            Memory(
                id=3,
                content="我对海鲜过敏，不能吃螃蟹和虾",
                summary="过敏信息",
                source_type="explicit",
                embedding=None
            )
        ]

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.query_agent.name, "QueryAgent")
        self.assertEqual(self.query_agent.top_k, 3)
        self.assertEqual(self.query_agent.relevance_threshold, 0.3)
        self.assertEqual(self.query_agent.temperature, 0.7)

    def test_extract_search_terms(self):
        """测试提取搜索词"""
        query = "我喜欢什么水果？"

        # Mock LLM响应
        self.mock_llm.chat.return_value = MagicMock(content='''
        {
            "search_terms": ["喜欢", "水果"],
            "entities": [],
            "query_intent": "用户想知道自己喜欢什么水果",
            "expanded_terms": ["爱吃", "偏好", "口味"]
        }
        ''')

        search_info = self.query_agent.extract_search_terms(query)

        # 验证结果
        self.assertIn("search_terms", search_info)
        self.assertIn("entities", search_info)
        self.assertIn("query_intent", search_info)
        self.assertIn("expanded_terms", search_info)

        # 验证LLM被调用
        self.mock_llm.chat.assert_called_once()

    def test_extract_search_terms_fallback(self):
        """测试提取搜索词失败时的fallback"""
        query = "测试查询"

        # Mock LLM抛出异常
        self.mock_llm.chat.side_effect = Exception("LLM error")

        search_info = self.query_agent.extract_search_terms(query)

        # 验证返回默认值
        self.assertEqual(search_info["search_terms"], ["测试查询"])
        self.assertEqual(search_info["entities"], [])
        self.assertEqual(search_info["query_intent"], "测试查询")
        self.assertEqual(search_info["expanded_terms"], [])

    def test_search_memories_by_terms(self):
        """测试通过关键词搜索记忆"""
        search_terms = ["喜欢", "水果"]
        entities = []

        # Mock搜索结果
        self.mock_mem_repo.search_by_text.return_value = [self.test_memories[0]]

        memories = self.query_agent.search_memories(search_terms, entities)

        # 验证结果
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].id, 1)

        # 验证repository被调用
        self.mock_mem_repo.search_by_text.assert_any_call("喜欢", limit=3)
        self.mock_mem_repo.search_by_text.assert_any_call("水果", limit=3)

    def test_search_memories_by_entities(self):
        """测试通过实体搜索记忆"""
        search_terms = []
        entities = ["苹果", "香蕉"]

        # Mock搜索结果
        self.mock_mem_repo.search_by_entities.return_value = [self.test_memories[0]]

        memories = self.query_agent.search_memories(search_terms, entities)

        # 验证结果
        self.assertEqual(len(memories), 1)

        # 验证repository被调用
        self.mock_mem_repo.search_by_entities.assert_any_call(
            ["苹果"],
            limit=3
        )
        self.mock_mem_repo.search_by_entities.assert_any_call(
            ["香蕉"],
            limit=3
        )

    def test_search_memories_with_vector_search(self):
        """测试向量搜索"""
        search_terms = ["测试"]
        entities = []

        # 启用向量搜索
        self.query_agent.set_config("enable_vector_search", True)
        self.mock_llm.supports_embedding.return_value = True

        # Mock embedding响应
        mock_embedding = Mock()
        mock_embedding.embeddings = [[0.1, 0.2, 0.3]]
        self.mock_llm.embed.return_value = mock_embedding
        
        # Mock search_by_text to return empty list
        self.mock_mem_repo.search_by_text.return_value = []

        # Mock向量搜索结果
        self.mock_mem_repo.search_by_vector.return_value = [(self.test_memories[0], 0.9)]

        memories = self.query_agent.search_memories(search_terms, entities)

        # 验证embedding被调用
        self.mock_llm.embed.assert_called_once()

    def test_search_memories_deduplication(self):
        """测试搜索结果去重"""
        search_terms = ["喜欢"]
        entities = []

        # Mock返回重复的记忆
        self.mock_mem_repo.search_by_text.return_value = self.test_memories
        self.mock_mem_repo.search_by_entities.return_value = self.test_memories

        memories = self.query_agent.search_memories(search_terms, entities)

        # 验证去重
        memory_ids = [m.id for m in memories]
        self.assertEqual(len(memory_ids), len(set(memory_ids)))

    def test_analyze_retrieved_memories(self):
        """测试分析检索到的记忆"""
        query = "我喜欢什么水果？"

        # Mock LLM响应
        self.mock_llm.chat.return_value = MagicMock(content='''
        {
            "relevant_memories": [
                {
                    "memory_id": 1,
                    "relevance_score": 0.95,
                    "summary": "直接回答了问题"
                }
            ],
            "total_count": 3,
            "relevant_count": 1
        }
        ''')

        result = self.query_agent.analyze_retrieved_memories(
            query,
            self.test_memories
        )

        # 验证结果
        self.assertEqual(result["relevant_count"], 1)
        self.assertEqual(len(result["relevant_memories"]), 1)
        self.assertEqual(result["relevant_memories"][0].id, 1)

    def test_analyze_retrieved_memories_empty(self):
        """测试分析空记忆列表"""
        query = "测试查询"
        memories = []

        result = self.query_agent.analyze_retrieved_memories(query, memories)

        # 验证结果
        self.assertEqual(result["relevant_memories"], [])
        self.assertEqual(result["total_count"], 0)
        self.assertEqual(result["relevant_count"], 0)

    def test_analyze_retrieved_memories_fallback(self):
        """测试分析失败时的fallback"""
        query = "测试查询"

        # Mock LLM抛出异常
        self.mock_llm.chat.side_effect = Exception("LLM error")

        result = self.query_agent.analyze_retrieved_memories(
            query,
            self.test_memories
        )

        # 验证返回所有记忆
        self.assertEqual(len(result["relevant_memories"]), 3)
        self.assertEqual(result["total_count"], 3)

    def test_synthesize_answer(self):
        """测试生成答案"""
        query = "我喜欢什么水果？"

        # Mock LLM响应
        self.mock_llm.chat.return_value = MagicMock(content="根据记忆，你喜欢苹果和香蕉。")

        answer = self.query_agent.synthesize_answer(
            query,
            [self.test_memories[0]]
        )

        # 验证结果
        self.assertEqual(answer, "根据记忆，你喜欢苹果和香蕉。")
        self.mock_llm.chat.assert_called_once()

    def test_synthesize_answer_empty_memories(self):
        """测试生成答案时没有记忆"""
        query = "测试查询"
        memories = []

        answer = self.query_agent.synthesize_answer(query, memories)

        # 验证返回空字符串
        self.assertEqual(answer, "")

    def test_generate_summary(self):
        """测试生成摘要"""
        query = "我的饮食偏好"

        # Mock LLM响应
        self.mock_llm.chat.return_value = MagicMock(content="你喜欢吃水果，特别是苹果和香蕉，但对海鲜过敏。")

        summary = self.query_agent.generate_summary(
            query,
            self.test_memories
        )

        # 验证结果
        self.assertEqual(
            summary,
            "你喜欢吃水果，特别是苹果和香蕉，但对海鲜过敏。"
        )

    def test_handle_conversation(self):
        """测试处理对话上下文"""
        history = [
            {"role": "user", "content": "我今天吃了苹果"},
            {"role": "assistant", "content": "好的，记录下来了"}
        ]
        current_query = "我还吃了什么？"

        # Mock LLM响应
        self.mock_llm.chat.return_value = MagicMock(content='''
        {
            "context": "用户在讨论今天吃的水果",
            "intent": "用户想知道今天还吃了什么水果",
            "missing_info": []
        }
        ''')

        result = self.query_agent.handle_conversation(history, current_query)

        # 验证结果
        self.assertIn("context", result)
        self.assertIn("intent", result)
        self.assertIn("missing_info", result)

    def test_handle_conversation_disabled(self):
        """测试禁用对话功能"""
        self.query_agent.set_config("enable_conversation", False)

        history = [{"role": "user", "content": "测试"}]
        current_query = "当前问题"

        result = self.query_agent.handle_conversation(history, current_query)

        # 验证返回默认值
        self.assertEqual(result["context"], "")
        self.assertEqual(result["intent"], current_query)

    def test_process_search_mode(self):
        """测试处理查询（搜索模式）"""
        query = "我的水果喜好"

        # Mock搜索结果
        self.mock_mem_repo.search_by_text.return_value = [self.test_memories[0]]

        # Mock分析结果
        self.mock_llm.chat.return_value = MagicMock(content='''
        {
            "relevant_memories": [
                {
                    "memory_id": 1,
                    "relevance_score": 0.95,
                    "summary": "相关"
                }
            ],
            "total_count": 1,
            "relevant_count": 1
        }
        ''')

        result = self.query_agent.process(query, mode="search")

        # 验证结果
        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "search")
        self.assertIn("answer", result)

    def test_process_answer_mode(self):
        """测试处理查询（答案模式）"""
        query = "我喜欢什么水果？"

        # Mock搜索结果
        self.mock_mem_repo.search_by_text.return_value = [self.test_memories[0]]

        # Mock分析结果
        import json
        self.mock_llm.chat.side_effect = [
            # extract_search_terms
            MagicMock(content=json.dumps({
                "search_terms": ["水果"],
                "entities": [],
                "query_intent": "查询",
                "expanded_terms": []
            })),
            # analyze_retrieved_memories
            MagicMock(content=json.dumps({
                "relevant_memories": [
                    {"memory_id": 1, "relevance_score": 0.95, "summary": "相关"}
                ],
                "total_count": 1,
                "relevant_count": 1
            })),
            # synthesize_answer
            MagicMock(content="你喜欢苹果和香蕉。")
        ]

        result = self.query_agent.process(query, mode="answer")

        # 验证结果
        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "answer")
        self.assertEqual(result["answer"], "你喜欢苹果和香蕉。")

    def test_process_no_results(self):
        """测试处理查询（无结果）"""
        query = "我不喜欢吃什么？"

        # Mock空搜索结果
        self.mock_mem_repo.search_by_text.return_value = []

        # Mock无结果处理
        self.mock_llm.chat.return_value = MagicMock(content="抱歉，没有找到相关的记忆。")

        result = self.query_agent.process(query)

        # 验证结果
        self.assertTrue(result["success"])
        self.assertEqual(result["relevant_memories"], [])
        self.assertEqual(result["confidence"], 0.0)

    def test_clear_conversation(self):
        """测试清除对话历史"""
        self.query_agent.conversation_history = [
            {"role": "user", "content": "测试"}
        ]

        self.query_agent.clear_conversation()

        # 验证历史被清除
        self.assertEqual(len(self.query_agent.conversation_history), 0)

    def test_get_conversation_history(self):
        """测试获取对话历史"""
        history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "答案1"}
        ]
        self.query_agent.conversation_history = history

        result = self.query_agent.get_conversation_history()

        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["content"], "问题1")


if __name__ == '__main__':
    # 需要导入json模块用于测试
    import json
    unittest.main()
