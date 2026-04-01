"""
IngestAgent（摄入智能体）
负责处理用户输入的信息，包括总结、实体提取、主题打标和向量存储
"""

import json
import os
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from memory_system.agents.base_agent import BaseAgent
from memory_system.agents.ingest_prompts import (
    get_summarize_prompt,
    get_extract_entities_prompt,
    get_tag_topics_prompt,
    get_assess_importance_prompt
)
from memory_system.llm import LLMProvider, create_llm_provider
from memory_system.storage import Memory, MemoryRepository, FileRepository
from memory_system.storage.database import DatabaseManager
from memory_system.utils.logger import get_agent_logger
from memory_system.utils.json_utils import extract_json


class IngestAgent(BaseAgent):
    """
    摄入智能体

    负责将用户输入的文本或文件转化为结构化的记忆
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        db_manager: DatabaseManager,
        embedding_provider: Optional[LLMProvider] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化IngestAgent

        Args:
            llm_provider: 用于文本处理的LLM提供商
            db_manager: 数据库管理器
            embedding_provider: 用于向量化的LLM提供商（可选，默认使用llm_provider）
            config: 配置参数
        """
        system_prompt = """你是一个专业的信息处理助手，负责对用户输入的内容进行总结、实体提取和主题打标。请准确地完成任务，并以指定格式输出结果。"""

        super().__init__(
            name="IngestAgent",
            llm_provider=llm_provider,
            system_prompt=system_prompt,
            config=config
        )

        self.db_manager = db_manager
        self.embedding_provider = embedding_provider or llm_provider

        # 确保embedding提供商支持embedding
        if not self.embedding_provider.supports_embedding():
            raise ValueError(f"{self.embedding_provider.__class__.__name__} 不支持embedding功能")

    def process(
        self,
        content: str,
        source_type: str = "user_input",
        source_id: Optional[int] = None
    ) -> Memory:
        """
        处理内容并创建记忆

        Args:
            content: 要处理的内容
            source_type: 来源类型
            source_id: 来源ID（如文件ID）

        Returns:
            Memory: 创建的记忆对象
        """
        self.logger.info(f"开始处理内容: 类型={source_type}, 长度={len(content)}")

        # 1. 总结
        summary = self.summarize_content(content)

        # 2. 提取实体
        entities = self.extract_entities(content)

        # 3. 打主题标签
        topics = self.tag_topics(content)

        # 4. 评估重要性
        importance_score = self.assess_importance(content)

        # 5. 向量化
        embedding = self.create_embedding(content)

        # 6. 存储到数据库
        memory = self._store_memory(
            content=content,
            summary=summary,
            entities=entities,
            topics=topics,
            embedding=embedding,
            source_type=source_type,
            source_id=source_id,
            importance_score=importance_score
        )

        self.logger.info(f"记忆创建成功: ID={memory.id}")
        return memory

    def summarize_content(self, content: str) -> str:
        """
        生成内容摘要

        Args:
            content: 原始内容

        Returns:
            str: 摘要
        """
        prompt = get_summarize_prompt(content)
        summary = self.chat(prompt, temperature=0.5)
        return summary.strip()

    def extract_entities(self, content: str) -> List[str]:
        """
        提取实体

        Args:
            content: 原始内容

        Returns:
            List[str]: 实体列表
        """
        prompt = get_extract_entities_prompt(content)
        response = self.chat(prompt, temperature=0.3)

        entities = extract_json(response)
        if isinstance(entities, list):
            return [str(e) for e in entities]
        else:
            self.logger.warning(f"实体提取返回非列表格式: {response}")
            return []

    def tag_topics(self, content: str) -> List[str]:
        """
        打主题标签

        Args:
            content: 原始内容

        Returns:
            List[str]: 主题标签列表
        """
        prompt = get_tag_topics_prompt(content)
        response = self.chat(prompt, temperature=0.3)

        topics = extract_json(response)
        if isinstance(topics, list):
            return [str(t) for t in topics]
        else:
            self.logger.warning(f"主题打标返回非列表格式: {response}")
            return []

    def assess_importance(self, content: str) -> float:
        """
        评估内容的重要性

        Args:
            content: 原始内容

        Returns:
            float: 重要性评分 (0-1)
        """
        prompt = get_assess_importance_prompt(content)
        response = self.chat(prompt, temperature=0.3)

        try:
            score = float(response.strip())
            # 确保分数在有效范围内
            return max(0.0, min(1.0, score))
        except ValueError:
            self.logger.warning(f"重要性评估返回非数字格式: {response}")
            return 0.5  # 默认中等重要性

    def create_embedding(self, content: str) -> List[float]:
        """
        创建向量

        Args:
            content: 要向量化的内容

        Returns:
            List[float]: 向量
        """
        try:
            from memory_system.llm import EmbeddingResponse

            # 使用摘要内容进行向量化（更简洁）
            text_to_embed = content[:2000] if len(content) > 2000 else content

            response = self.embedding_provider.embed(
                texts=[text_to_embed]
            )

            if isinstance(response, EmbeddingResponse) and len(response.embeddings) > 0:
                return response.embeddings[0]
            else:
                self.logger.warning("Embedding响应格式异常")
                return []

        except Exception as e:
            self.logger.error(f"创建embedding失败: {e}")
            return []

    def ingest_text(self, text: str) -> Memory:
        """
        摄入文本

        Args:
            text: 用户输入的文本

        Returns:
            Memory: 创建的记忆对象
        """
        return self.process(content=text, source_type="user_input")

    def ingest_file(self, file_path: str) -> Memory:
        """
        摄入文件

        Args:
            file_path: 文件路径

        Returns:
            Memory: 创建的记忆对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取文件内容
        content, file_type, file_size = self._read_file(file_path)

        # 创建文件记录
        with self.db_manager.get_session() as session:
            file_repo = self.db_manager.get_file_repository(session)
            filename = os.path.basename(file_path)

            file_record = file_repo.create(
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                content=content
            )

            # 更新状态为处理中
            file_repo.update_status(file_record.id, "processing")

        try:
            # 处理内容并创建记忆
            memory = self.process(
                content=content,
                source_type="file_upload",
                source_id=file_record.id
            )

            # 更新文件状态为完成
            with self.db_manager.get_session() as session:
                file_repo = self.db_manager.get_file_repository(session)
                file_repo.update_status(
                    file_record.id,
                    "completed",
                    memory_id=memory.id
                )

            return memory

        except Exception as e:
            # 更新文件状态为失败
            with self.db_manager.get_session() as session:
                file_repo = self.db_manager.get_file_repository(session)
                file_repo.update_status(
                    file_record.id,
                    "failed",
                    error_message=str(e)
                )
            raise

    def _read_file(self, file_path: str) -> tuple:
        """
        读取文件内容

        Args:
            file_path: 文件路径

        Returns:
            tuple: (内容, 文件类型, 文件大小)
        """
        # 获取文件大小
        file_size = os.path.getsize(file_path)

        # 根据扩展名判断文件类型
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # 支持的文件类型
        if ext in ['.txt', '.md', '.markdown']:
            file_type = f"text/{ext[1:]}"
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        elif ext == '.json':
            file_type = 'application/json'
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content = json.dumps(data, ensure_ascii=False, indent=2)

        else:
            # 默认尝试作为文本读取
            file_type = 'text/plain'
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                raise ValueError(f"不支持的文件类型: {ext}")

        return content, file_type, file_size

    def _store_memory(
        self,
        content: str,
        summary: str,
        entities: List[str],
        topics: List[str],
        embedding: List[float],
        source_type: str,
        source_id: Optional[int],
        importance_score: float = 0.5
    ) -> Memory:
        """
        存储记忆到数据库

        Args:
            content: 内容
            summary: 摘要
            entities: 实体列表
            topics: 主题列表
            embedding: 向量
            source_type: 来源类型
            source_id: 来源ID
            importance_score: 重要性评分

        Returns:
            Memory: 创建的记忆对象
        """
        with self.db_manager.get_session() as session:
            mem_repo = self.db_manager.get_memory_repository(session)

            memory = mem_repo.create(
                content=content,
                summary=summary,
                entities=entities if entities else None,
                topics=topics if topics else None,
                embedding=embedding if embedding else None,
                source_type=source_type,
                source_id=source_id,
                importance_score=importance_score
            )

        return memory


def create_ingest_agent(
    db_manager: DatabaseManager,
    chat_provider_name: Optional[str] = None,
    embedding_provider_name: Optional[str] = None
) -> IngestAgent:
    """
    创建IngestAgent实例（便捷函数）

    Args:
        db_manager: 数据库管理器
        chat_provider_name: 聊天提供商名称
        embedding_provider_name: 向量化提供商名称

    Returns:
        IngestAgent: IngestAgent实例
    """
    from memory_system.config.settings import get_settings

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

    # 创建embedding提供商
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

    return IngestAgent(
        llm_provider=chat_provider,
        db_manager=db_manager,
        embedding_provider=embedding_provider
    )
