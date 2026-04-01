"""
查询智能体
负责根据用户的查询从记忆库中搜索相关信息，并生成准确的答案
"""

import json
from typing import List, Optional, Dict, Any
from memory_system.agents.base_agent import BaseAgent
from memory_system.agents.query_prompts import (
    EXTRACT_SEARCH_TERMS_PROMPT,
    ANALYZE_RETRIEVED_MEMORIES_PROMPT,
    SYNTHESIZE_ANSWER_PROMPT,
    GENERATE_SUMMARY_PROMPT,
    CONVERSATION_CONTEXT_PROMPT,
    NO_RESULTS_PROMPT,
    CONFIDENCE_ASSESSMENT_PROMPT
)
from memory_system.storage.models import Memory
from memory_system.utils.logger import get_agent_logger
from memory_system.utils.json_utils import extract_json


class QueryAgent(BaseAgent):
    """
    查询智能体

    功能：
    1. 理解用户查询意图
    2. 从记忆库中搜索相关信息
    3. 分析检索结果的相关性
    4. 基于相关记忆生成准确的答案
    5. 支持多轮对话上下文
    """

    def __init__(
        self,
        llm_provider,
        db_manager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化查询智能体

        Args:
            llm_provider: LLM提供商
            db_manager: 数据库管理器
            config: 配置参数（优先）或使用默认配置
        """
        # 配置默认值
        default_config = {
            "top_k": 5,                      # 返回的记忆数量
            "relevance_threshold": 0.3,      # 相关度阈值
            "max_context_length": 4000,      # 最大上下文长度
            "enable_vector_search": True,   # 是否启用向量搜索
            "enable_conversation": True,     # 是否支持多轮对话
            "temperature": 0.7,              # 温度参数
            "max_tokens": 1000               # 最大token数
        }

        if config:
            default_config.update(config)

        super().__init__(
            name="QueryAgent",
            llm_provider=llm_provider,
            config=default_config
        )

        self.db_manager = db_manager
        self.conversation_history: List[Dict[str, str]] = []
        self.logger = get_agent_logger()

        self.logger.info("QueryAgent 初始化完成")

    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        处理用户查询（主入口）

        Args:
            query: 用户查询
            **kwargs: 其他参数（如mode="search"|"answer"|"summary"）

        Returns:
            Dict: 处理结果
            {
                "success": True,
                "query": "用户查询",
                "answer": "生成的答案",
                "relevant_memories": [...],
                "confidence": 0.95,
                "mode": "answer"
            }
        """
        mode = kwargs.get("mode", "answer")  # search, answer, summary

        self.logger.info(f"处理查询: {query} (mode: {mode})")

        try:
            # 1. 提取搜索词
            search_info = self.extract_search_terms(query)

            # 2. 搜索记忆
            retrieved_memories = self.search_memories(
                search_info["search_terms"],
                search_info["entities"]
            )

            # 3. 分析检索结果
            analyzed_result = self.analyze_retrieved_memories(
                query,
                retrieved_memories
            )

            relevant_memories = analyzed_result["relevant_memories"]

            # 4. 根据模式处理
            if not relevant_memories:
                # 没有找到相关记忆
                answer = self._handle_no_results(query)
                confidence = 0.0
            elif mode == "search":
                # 只返回搜索结果
                answer = self._format_search_results(relevant_memories)
                confidence = 1.0
            elif mode == "summary":
                # 生成摘要
                answer = self.generate_summary(query, relevant_memories)
                confidence = self._assess_confidence(
                    query, answer, relevant_memories
                )
            else:  # mode == "answer"
                # 生成答案
                answer = self.synthesize_answer(query, relevant_memories)
                confidence = self._assess_confidence(
                    query, answer, relevant_memories
                )

            # 5. 更新对话历史
            self._update_conversation_history(query, answer)

            result = {
                "success": True,
                "query": query,
                "answer": answer,
                "relevant_memories": relevant_memories,
                "confidence": confidence,
                "mode": mode
            }

            self.logger.info(f"查询处理完成: confidence={confidence}")
            return result

        except Exception as e:
            self.logger.error(f"查询处理失败: {e}")
            return {
                "success": False,
                "query": query,
                "answer": f"抱歉，处理您的查询时出错: {str(e)}",
                "relevant_memories": [],
                "confidence": 0.0,
                "mode": mode,
                "error": str(e)
            }

    def extract_search_terms(self, query: str) -> Dict[str, Any]:
        """
        从用户查询中提取搜索词和实体

        Args:
            query: 用户查询

        Returns:
            Dict: 搜索信息
            {
                "search_terms": [...],
                "entities": [...],
                "query_intent": "...",
                "expanded_terms": [...]
            }
        """
        prompt = EXTRACT_SEARCH_TERMS_PROMPT.format(query=query)

        response = self.chat(prompt, temperature=0.3)
        search_info = extract_json(response)
        
        if not isinstance(search_info, dict):
            self.logger.error(f"提取搜索词失败 (Response: {response})")
            return {
                "search_terms": query.split(),
                "entities": [],
                "query_intent": query,
                "expanded_terms": []
            }

        # 验证返回的数据结构
        required_keys = ["search_terms", "entities", "query_intent", "expanded_terms"]
        for key in required_keys:
            if key not in search_info:
                search_info[key] = []

        self.logger.debug(f"提取搜索词: {search_info['search_terms']}")
        return search_info

    def search_memories(
        self,
        search_terms: List[str],
        entities: List[str]
    ) -> List[Memory]:
        """
        根据搜索词和实体搜索记忆

        Args:
            search_terms: 搜索词列表
            entities: 实体列表

        Returns:
            List[Memory]: 检索到的记忆列表
        """
        with self.db_manager.get_session() as session:
            mem_repo = self.db_manager.get_memory_repository(session)
            rel_repo = self.db_manager.get_relation_repository(session)

            memories = []

            # 1. 关键词搜索
            for term in search_terms:
                results = mem_repo.search_by_text(term, limit=self.top_k)
                memories.extend(results)

            # 2. 实体搜索
            for entity in entities:
                results = mem_repo.search_by_entities([entity], limit=self.top_k)
                memories.extend(results)

            # 3. 向量搜索（如果支持）
            if self.enable_vector_search and self.llm_provider.supports_embedding():
                query_text = " ".join(search_terms + entities)
                embedding_response = self.llm_provider.embed([query_text])

                if embedding_response.embeddings:
                    query_vector = embedding_response.embeddings[0]
                    results = mem_repo.search_by_vector(
                        query_vector,
                        top_k=self.top_k
                    )
                    # Extract Memory objects from (Memory, float) tuples
                    results = [r[0] for r in results]
                    memories.extend(results)

            # 去重
            unique_memories = []
            seen_ids = set()

            for mem in memories:
                if mem.id not in seen_ids:
                    unique_memories.append(mem)
                    seen_ids.add(mem.id)

            # 限制数量
            return unique_memories[:self.top_k]

    def analyze_retrieved_memories(
        self,
        query: str,
        memories: List[Memory]
    ) -> Dict[str, Any]:
        """
        分析检索到的记忆的相关性

        Args:
            query: 用户查询
            memories: 检索到的记忆列表

        Returns:
            Dict: 分析结果
            {
                "relevant_memories": [...],
                "total_count": 5,
                "relevant_count": 3
            }
        """
        if not memories:
            return {
                "relevant_memories": [],
                "total_count": 0,
                "relevant_count": 0
            }

        # 格式化记忆列表
        memories_str = ""
        for i, mem in enumerate(memories, 1):
            memories_str += f"\n记忆ID={mem.id}\n"
            memories_str += f"内容: {mem.content}\n"
            memories_str += f"摘要: {mem.summary}\n"

        prompt = ANALYZE_RETRIEVED_MEMORIES_PROMPT.format(
            query=query,
            memories=memories_str
        )

        response = self.chat(prompt, temperature=0.3)
        analysis = extract_json(response)

        if not isinstance(analysis, dict):
            self.logger.error(f"分析记忆相关性失败: {response}")
            # 返回所有记忆
            return {
                "relevant_memories": memories,
                "total_count": len(memories),
                "relevant_count": len(memories)
            }

        # 获取相关记忆
        relevant_memories = []
        for mem_data in analysis.get("relevant_memories", []):
            mem_id = mem_data["memory_id"]
            matched_mem = next((m for m in memories if m.id == mem_id), None)
            if matched_mem and mem_data["relevance_score"] >= self.relevance_threshold:
                relevant_memories.append(matched_mem)

        self.logger.debug(f"相关记忆数量: {len(relevant_memories)}/{len(memories)}")

        return {
            "relevant_memories": relevant_memories,
            "total_count": len(memories),
            "relevant_count": len(relevant_memories)
        }

    def synthesize_answer(
        self,
        query: str,
        memories: List[Memory]
    ) -> str:
        """
        基于相关记忆生成答案

        Args:
            query: 用户查询
            memories: 相关记忆列表

        Returns:
            str: 生成的答案
        """
        if not memories:
            return ""

        # 格式化记忆内容
        memories_str = ""
        for i, mem in enumerate(memories, 1):
            memories_str += f"\n记忆{i}:\n"
            memories_str += f"- 摘要: {mem.summary}\n"
            memories_str += f"- 内容: {mem.content}\n"

        prompt = SYNTHESIZE_ANSWER_PROMPT.format(
            query=query,
            memories=memories_str
        )

        response = self.chat(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response

    def generate_summary(
        self,
        query: str,
        memories: List[Memory]
    ) -> str:
        """
        生成主题摘要

        Args:
            query: 用户查询
            memories: 相关记忆列表

        Returns:
            str: 生成的摘要
        """
        if not memories:
            return ""

        # 格式化记忆内容
        memories_str = ""
        for i, mem in enumerate(memories, 1):
            memories_str += f"\n记忆{i}:\n"
            memories_str += f"- 摘要: {mem.summary}\n"
            memories_str += f"- 内容: {mem.content}\n"

        prompt = GENERATE_SUMMARY_PROMPT.format(
            query=query,
            memories=memories_str
        )

        response = self.chat(
            prompt,
            temperature=0.5,
            max_tokens=self.max_tokens
        )

        return response

    def handle_conversation(
        self,
        history: List[Dict[str, str]],
        current_query: str
    ) -> Dict[str, Any]:
        """
        处理多轮对话上下文

        Args:
            history: 对话历史 [{"role": "user", "content": "..."}, ...]
            current_query: 当前查询

        Returns:
            Dict: 上下文分析结果
            {
                "context": "...",
                "intent": "...",
                "missing_info": [...]
            }
        """
        if not self.enable_conversation:
            return {
                "context": "",
                "intent": current_query,
                "missing_info": []
            }

        history_str = ""
        for msg in history[-5:]:  # 只保留最近5轮
            role = "用户" if msg["role"] == "user" else "助手"
            history_str += f"{role}: {msg['content']}\n"

        prompt = CONVERSATION_CONTEXT_PROMPT.format(
            history=history_str,
            query=current_query
        )

        response = self.chat(prompt, temperature=0.3)
        context_info = extract_json(response)
        if isinstance(context_info, dict):
            return context_info
        else:
            self.logger.error(f"分析对话上下文失败: {response}")
            return {
                "context": "",
                "intent": current_query,
                "missing_info": []
            }

    def _handle_no_results(self, query: str) -> str:
        """处理无搜索结果的情况"""
        prompt = NO_RESULTS_PROMPT.format(query=query)
        return self.chat(prompt, temperature=0.7)

    def _format_search_results(self, memories: List[Memory]) -> str:
        """格式化搜索结果"""
        if not memories:
            return "没有找到相关的记忆。"

        result = f"找到 {len(memories)} 条相关记忆：\n\n"
        for i, mem in enumerate(memories, 1):
            result += f"{i}. {mem.summary}\n"
            result += f"   {mem.content[:100]}...\n\n"

        return result

    def _assess_confidence(
        self,
        query: str,
        answer: str,
        memories: List[Memory]
    ) -> float:
        """评估答案置信度"""
        if not memories:
            return 0.0

        # 格式化记忆
        memories_str = ""
        for mem in memories[:3]:
            memories_str += f"- {mem.summary}: {mem.content}\n"

        prompt = CONFIDENCE_ASSESSMENT_PROMPT.format(
            query=query,
            answer=answer,
            memories=memories_str
        )

        response = self.chat(prompt, temperature=0.3)
        assessment = extract_json(response)
        if isinstance(assessment, dict):
            return assessment.get("confidence_score", 0.5)
        else:
            self.logger.error(f"评估置信度失败: {response}")
            return 0.5

    def _update_conversation_history(self, query: str, answer: str):
        """更新对话历史"""
        self.conversation_history.append({
            "role": "user",
            "content": query
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        # 限制历史长度
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def clear_conversation(self):
        """清除对话历史"""
        self.conversation_history = []
        self.logger.info("对话历史已清除")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history

    @property
    def top_k(self) -> int:
        """返回的记忆数量"""
        return self.get_config("top_k", 5)

    @property
    def relevance_threshold(self) -> float:
        """相关度阈值"""
        return self.get_config("relevance_threshold", 0.3)

    @property
    def max_context_length(self) -> int:
        """最大上下文长度"""
        return self.get_config("max_context_length", 4000)

    @property
    def enable_vector_search(self) -> bool:
        """是否启用向量搜索"""
        return self.get_config("enable_vector_search", True)

    @property
    def enable_conversation(self) -> bool:
        """是否支持多轮对话"""
        return self.get_config("enable_conversation", True)

    @property
    def temperature(self) -> float:
        """温度参数"""
        return self.get_config("temperature", 0.7)

    @property
    def max_tokens(self) -> int:
        """最大token数"""
        return self.get_config("max_tokens", 1000)
