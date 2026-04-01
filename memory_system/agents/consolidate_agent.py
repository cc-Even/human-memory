"""
ConsolidateAgent（整合智能体）
负责发现记忆之间的关联、整合相似记忆
"""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone

from memory_system.agents.base_agent import BaseAgent
from memory_system.agents.consolidate_prompts import (
    get_generate_keywords_prompt,
    get_judge_related_prompt,
    get_merge_memories_prompt
)
from memory_system.llm import LLMProvider
from memory_system.storage import Memory, MemoryRepository, RelationRepository
from memory_system.storage.database import DatabaseManager
from memory_system.utils.logger import get_agent_logger
from memory_system.utils.json_utils import extract_json


class ConsolidateAgent(BaseAgent):
    """
    整合智能体

    负责分析记忆之间的关联关系，整合相似记忆，构建记忆网络
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        db_manager: DatabaseManager,
        embedding_provider: Optional[LLMProvider] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化ConsolidateAgent

        Args:
            llm_provider: 用于文本处理的LLM提供商
            db_manager: 数据库管理器
            embedding_provider: 用于向量化的LLM提供商（可选）
            config: 配置参数
        """
        system_prompt = """你是一个专业的记忆整合助手，负责发现记忆之间的关联关系，分析并整合相似的记忆。请准确地完成任务，并以指定格式输出结果。"""

        super().__init__(
            name="ConsolidateAgent",
            llm_provider=llm_provider,
            system_prompt=system_prompt,
            config=config
        )

        self.db_manager = db_manager
        self.embedding_provider = embedding_provider or llm_provider

        # 默认配置（可被config覆盖）
        self.similarity_threshold = self.get_config("similarity_threshold", 0.75)
        self.top_k = self.get_config("top_k", 5)
        self.max_merge_count = self.get_config("max_merge_count", 3)
        # 自动整合配置
        self.consolidation_interval_hours = self.get_config("consolidation_interval_hours", 24)
        self.auto_consolidate_enabled = self.get_config("auto_consolidate_enabled", False)

    def process(self, memory_id: int) -> Dict[str, Any]:
        """
        处理单个记忆，发现关联并整合

        Args:
            memory_id: 要处理的记忆ID

        Returns:
            Dict[str, Any]: 处理结果统计
        """
        self.logger.info(f"开始处理记忆 ID={memory_id}")

        # 获取记忆
        with self.db_manager.get_session() as session:
            mem_repo = self.db_manager.get_memory_repository(session)
            memory = mem_repo.get_by_id(memory_id)

            if not memory:
                self.logger.warning(f"记忆不存在: ID={memory_id}")
                return {"success": False, "reason": "Memory not found"}

        # 生成关键词
        keywords = self.generate_keywords(memory.content)

        # 发现相关记忆
        related_memories = self.find_related_memories(
            keywords=keywords,
            exclude_id=memory_id
        )

        # 判断关联并建立关系
        relations_created = 0
        for related_memory in related_memories:
            relation = self.judge_related(memory, related_memory)

            if relation["related"] and relation["score"] >= self.similarity_threshold:
                self._create_relation(
                    memory.id,
                    related_memory.id,
                    relation_type="similar",
                    similarity_score=relation["score"],
                    reason=relation["reason"]
                )
                relations_created += 1

        self.logger.info(f"处理完成: 关系数={relations_created}")

        return {
            "success": True,
            "memory_id": memory_id,
            "keywords": keywords,
            "related_count": len(related_memories),
            "relations_created": relations_created
        }

    def consolidate(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        批量整合指定时间窗口内的记忆

        Args:
            time_window_hours: 时间窗口（小时）

        Returns:
            Dict[str, Any]: 整合结果统计
        """
        self.logger.info(f"开始批量整合: 时间窗口={time_window_hours}小时")

        # 获取时间窗口内的记忆
        time_threshold = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=time_window_hours)

        with self.db_manager.get_session() as session:
            mem_repo = self.db_manager.get_memory_repository(session)
            memories = mem_repo.search_by_date_range(
                start_date=time_threshold,
                end_date=datetime.now(timezone.utc).replace(tzinfo=None)
            )

        # 处理每个记忆
        results = {
            "total": len(memories),
            "processed": 0,
            "relations_created": 0,
            "merged": 0
        }

        for memory in memories:
            result = self.process(memory.id)

            if result["success"]:
                results["processed"] += 1
                results["relations_created"] += result["relations_created"]

        self.logger.info(f"批量整合完成: {results}")

        return results

    def find_patterns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        发现记忆规律和模式

        Args:
            limit: 最多分析的记忆数量

        Returns:
            List[Dict[str, Any]]: 发现的规律列表
        """
        self.logger.info(f"开始发现规律: 限制={limit}")

        # 获取最近的记忆
        with self.db_manager.get_session() as session:
            mem_repo = self.db_manager.get_memory_repository(session)
            memories = mem_repo.get_all(limit=limit)

        # 分析实体和主题的频率
        entity_counts: Dict[str, int] = {}
        topic_counts: Dict[str, int] = {}

        for memory in memories:
            if memory.entities:
                for entity in memory.entities:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1

            if memory.topics:
                for topic in memory.topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # 提取高频实体和主题
        patterns = []

        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for entity, count in top_entities:
            if count >= 2:  # 至少出现2次
                patterns.append({
                    "type": "entity",
                    "name": entity,
                    "frequency": count
                })

        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for topic, count in top_topics:
            if count >= 2:  # 至少出现2次
                patterns.append({
                    "type": "topic",
                    "name": topic,
                    "frequency": count
                })

        self.logger.info(f"发现规律: 数量={len(patterns)}")

        return patterns

    def merge_memories(self, memory_ids: List[int]) -> Optional[Memory]:
        """
        合并多个记忆

        Args:
            memory_ids: 要合并的记忆ID列表

        Returns:
            Optional[Memory]: 合并后的新记忆
        """
        self.logger.info(f"开始合并记忆: {memory_ids}")

        if len(memory_ids) < 2:
            self.logger.warning("至少需要2个记忆才能合并")
            return None

        if len(memory_ids) > self.max_merge_count:
            self.logger.warning(f"合并数量超过限制: {len(memory_ids)} > {self.max_merge_count}")
            return None

        # 获取记忆
        with self.db_manager.get_session() as session:
            mem_repo = self.db_manager.get_memory_repository(session)

            memories = []
            for mid in memory_ids:
                memory = mem_repo.get_by_id(mid)
                if memory:
                    memories.append(memory)

            if len(memories) < 2:
                self.logger.warning("有效记忆数量不足")
                return None

        # LLM合并
        merge_result = self._llm_merge(memories)

        if not merge_result:
            self.logger.warning("记忆合并失败")
            return None

        # 创建新记忆
        with self.db_manager.get_session() as session:
            mem_repo = self.db_manager.get_memory_repository(session)
            rel_repo = self.db_manager.get_relation_repository(session)

            # 创建新记忆
            # 合并记忆的重要性取原记忆中最高的
            max_importance = max(m.importance_score for m in memories) if memories else 0.5
            new_memory = mem_repo.create(
                content=merge_result["content"],
                summary=merge_result["summary"],
                entities=merge_result.get("entities"),
                topics=merge_result.get("topics"),
                source_type="consolidation",
                importance_score=max_importance
            )

            # 重新创建向量
            if self.embedding_provider and self.embedding_provider.supports_embedding():
                embedding = self.create_embedding(new_memory.content)
                mem_repo.update(new_memory.id, embedding=embedding)

            # 标记原记忆为已合并
            for memory in memories:
                mem_repo.update(memory.id, is_merged=True, merged_into=new_memory.id)

            # 创建关联关系
            for memory in memories:
                rel_repo.create(
                    memory_id=memory.id,
                    related_memory_id=new_memory.id,
                    relation_type="merged_to"
                )

        self.logger.info(f"合并完成: 新记忆ID={new_memory.id}")

        return new_memory

    def generate_keywords(self, text: str) -> List[str]:
        """
        生成搜索关键词

        Args:
            text: 记忆文本

        Returns:
            List[str]: 关键词列表
        """
        prompt = get_generate_keywords_prompt(text)
        response = self.chat(prompt, temperature=0.3)

        keywords = extract_json(response)
        if isinstance(keywords, list):
            return [str(k) for k in keywords]
        else:
            self.logger.warning(f"关键词提取失败或返回非列表格式: {response}")
            return []

    def find_related_memories(
        self,
        keywords: List[str],
        exclude_id: Optional[int] = None,
        limit: int = None
    ) -> List[Memory]:
        """
        查找相关记忆

        Args:
            keywords: 关键词列表
            exclude_id: 排除的记忆ID
            limit: 最多返回数量

        Returns:
            List[Memory]: 相关记忆列表
        """
        if not keywords:
            return []

        with self.db_manager.get_session() as session:
            mem_repo = self.db_manager.get_memory_repository(session)

            # 使用关键词搜索
            memories = []
            seen_ids = set()

            for keyword in keywords:
                results = mem_repo.search_by_text(keyword, limit=10)

                for memory in results:
                    if memory.id == exclude_id:
                        continue
                    if memory.id in seen_ids:
                        continue

                    memories.append(memory)
                    seen_ids.add(memory.id)

                    if limit and len(memories) >= limit:
                        break

                if limit and len(memories) >= limit:
                    break

        return memories[:limit] if limit else memories

    def judge_related(self, memory_a: Memory, memory_b: Memory) -> Dict[str, Any]:
        """
        判断两个记忆是否相关

        Args:
            memory_a: 记忆A
            memory_b: 记忆B

        Returns:
            Dict[str, Any]: 判断结果 {related: bool, score: float, reason: str}
        """
        prompt = get_judge_related_prompt(
            memory_a=memory_a.summary or memory_a.content[:500],
            memory_b=memory_b.summary or memory_b.content[:500]
        )
        response = self.chat(prompt, temperature=0.3)

        result = extract_json(response)
        if isinstance(result, dict):
            return {
                "related": result.get("related", False),
                "score": float(result.get("score", 0.0)),
                "reason": result.get("reason", "")
            }
        else:
            self.logger.warning(f"关联判断提取失败: {response}")
            return {"related": False, "score": 0.0, "reason": "Parse error"}

    def create_embedding(self, content: str) -> List[float]:
        """
        创建向量

        Args:
            content: 要向量化的内容

        Returns:
            List[float]: 向量
        """
        if not self.embedding_provider or not self.embedding_provider.supports_embedding():
            return []

        try:
            from memory_system.llm import EmbeddingResponse

            text_to_embed = content[:2000] if len(content) > 2000 else content

            response = self.embedding_provider.embed(texts=[text_to_embed])

            if isinstance(response, EmbeddingResponse) and len(response.embeddings) > 0:
                return response.embeddings[0]
            else:
                return []

        except Exception as e:
            self.logger.error(f"创建embedding失败: {e}")
            return []

    def _create_relation(
        self,
        memory_id: int,
        related_memory_id: int,
        relation_type: str,
        similarity_score: float,
        reason: str = ""
    ):
        """创建记忆关系"""
        with self.db_manager.get_session() as session:
            rel_repo = self.db_manager.get_relation_repository(session)

            # 检查是否已存在关系
            existing = rel_repo.get_relation(memory_id, related_memory_id)
            if existing:
                self.logger.debug(f"关系已存在: {memory_id} -> {related_memory_id}")
                return

            # 创建新关系
            rel_repo.create(
                memory_id=memory_id,
                related_memory_id=related_memory_id,
                relation_type=relation_type,
                similarity_score=similarity_score,
                notes=reason
            )

    def _llm_merge(self, memories: List[Memory]) -> Optional[Dict[str, Any]]:
        """
        使用LLM合并记忆

        Args:
            memories: 要合并的记忆列表

        Returns:
            Optional[Dict[str, Any]]: 合并结果
        """
        # 提取记忆内容
        memory_contents = [m.summary or m.content for m in memories]

        prompt = get_merge_memories_prompt(memory_contents)
        response = self.chat(prompt, temperature=0.5)

        result = extract_json(response)
        if isinstance(result, dict):
            return {
                "content": result.get("content", ""),
                "summary": result.get("summary", ""),
                "entities": result.get("entities", []),
                "topics": result.get("topics", [])
            }
        else:
            self.logger.warning(f"合并提取失败: {response}")
            return None


def create_consolidate_agent(
    db_manager: DatabaseManager,
    chat_provider_name: Optional[str] = None,
    embedding_provider_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> ConsolidateAgent:
    """
    创建ConsolidateAgent实例（便捷函数）

    Args:
        db_manager: 数据库管理器
        chat_provider_name: 聊天提供商名称
        embedding_provider_name: 向量化提供商名称（可选）
        config: 配置参数

    Returns:
        ConsolidateAgent: ConsolidateAgent实例
    """
    from memory_system.config.settings import get_settings
    from memory_system.llm import create_llm_provider

    settings = get_settings()
    llm_settings = settings.llm

    # 默认值
    chat_provider_name = chat_provider_name or llm_settings.default_llm_provider
    embedding_provider_name = embedding_provider_name or llm_settings.embedding_provider

    # 创建聊天提供商
    chat_kwargs = {}
    if chat_provider_name == "openai":
        chat_kwargs = {
            "api_key": llm_settings.openai_api_key,
            "base_url": llm_settings.openai_base_url,
            "chat_model": llm_settings.openai_chat_model,
            "embedding_model": llm_settings.openai_embedding_model
        }
    elif chat_provider_name == "dashscope":
        chat_kwargs = {
            "api_key": llm_settings.dashscope_api_key,
            "chat_model": llm_settings.dashscope_chat_model,
            "embedding_model": llm_settings.dashscope_embedding_model
        }
    elif chat_provider_name == "gemini":
        chat_kwargs = {
            "api_key": llm_settings.gemini_api_key,
            "model": llm_settings.gemini_chat_model
        }

    chat_provider = create_llm_provider(
        provider_name=chat_provider_name,
        api_key=chat_kwargs.pop("api_key"),
        **chat_kwargs
    )

    # 创建embedding提供商（如果指定）
    embedding_provider = None
    if embedding_provider_name:
        embed_kwargs = {}
        if embedding_provider_name == "openai":
            embed_kwargs = {
                "api_key": llm_settings.openai_api_key,
                "base_url": llm_settings.openai_base_url,
                "chat_model": llm_settings.openai_chat_model,
                "embedding_model": llm_settings.openai_embedding_model
            }
        elif embedding_provider_name == "dashscope":
            embed_kwargs = {
                "api_key": llm_settings.dashscope_api_key,
                "chat_model": llm_settings.dashscope_chat_model,
                "embedding_model": llm_settings.dashscope_embedding_model
            }
        elif embedding_provider_name == "gemini":
            embed_kwargs = {
                "api_key": llm_settings.gemini_api_key,
                "model": llm_settings.gemini_chat_model
            }

        embedding_provider = create_llm_provider(
            provider_name=embedding_provider_name,
            api_key=embed_kwargs.pop("api_key"),
            **embed_kwargs
        )

    return ConsolidateAgent(
        llm_provider=chat_provider,
        db_manager=db_manager,
        embedding_provider=embedding_provider,
        config=config
    )
