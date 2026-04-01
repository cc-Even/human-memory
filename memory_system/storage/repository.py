"""
数据库仓储层
提供对数据库操作的封装和高级查询接口
"""

from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
from datetime import datetime

from memory_system.storage.models import (
    Base,
    Memory,
    FileUpload,
    MemoryRelation
)
from memory_system.storage.vector_store import VectorStore
from memory_system.utils.logger import get_storage_logger


class BaseRepository:
    """基础仓储类"""

    def __init__(self, session: Session):
        """
        初始化仓储

        Args:
            session: 数据库会话
        """
        self.session = session
        self.logger = get_storage_logger()

    def add(self, entity):
        """
        添加实体

        Args:
            entity: 实体对象

        Returns:
            实体对象
        """
        self.session.add(entity)
        return entity

    def delete(self, entity):
        """
        删除实体

        Args:
            entity: 实体对象
        """
        self.session.delete(entity)

    def commit(self):
        """提交事务"""
        self.session.commit()

    def rollback(self):
        """回滚事务"""
        self.session.rollback()

    def close(self):
        """关闭会话"""
        self.session.close()


class MemoryRepository(BaseRepository):
    """记忆仓储类"""

    def __init__(self, session: Session, db_path: str):
        """
        初始化记忆仓储

        Args:
            session: 数据库会话
            db_path: 数据库文件路径（用于向量存储）
        """
        super().__init__(session)
        self.vector_store = VectorStore(db_path)

    def create(
        self,
        content: str,
        summary: Optional[str] = None,
        entities: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        source_type: str = "user_input",
        source_id: Optional[int] = None,
        importance_score: float = 0.5
    ) -> Memory:
        """
        创建新记忆

        Args:
            content: 原始内容
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
        # 转换embedding为JSON字符串
        embedding_str = None
        if embedding is not None:
            embedding_str = VectorStore.embedding_to_json(embedding)

        memory = Memory(
            content=content,
            summary=summary,
            entities=entities,
            topics=topics,
            embedding=embedding_str,
            source_type=source_type,
            source_id=source_id,
            importance_score=importance_score
        )

        self.add(memory)
        self.commit()

        self.logger.info(f"创建新记忆: ID={memory.id}, 内容长度={len(content)}")
        return memory


    def search_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100,
        offset: int = 0
    ) -> List[Memory]:
        """
        根据日期范围搜索记忆
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Memory]: 记忆列表
        """
        try:
            return self.session.query(Memory).filter(
                Memory.created_at >= start_date,
                Memory.created_at <= end_date
            ).order_by(Memory.created_at.desc()).limit(limit).offset(offset).all()
        except Exception as e:
            self.logger.error(f"按日期范围搜索失败: {str(e)}")
            return []

    def get_by_id(self, memory_id: int) -> Optional[Memory]:
        """
        根据ID获取记忆

        Args:
            memory_id: 记忆ID

        Returns:
            Optional[Memory]: 记忆对象或None
        """
        return self.session.query(Memory).filter(Memory.id == memory_id).first()

    def get_all(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: str = "created_at"
    ) -> List[Memory]:
        """
        获取所有记忆

        Args:
            limit: 限制数量
            offset: 偏移量
            order_by: 排序字段

        Returns:
            List[Memory]: 记忆列表
        """
        query = self.session.query(Memory)

        # 排序
        if order_by == "created_at":
            query = query.order_by(desc(Memory.created_at))
        elif order_by == "updated_at":
            query = query.order_by(desc(Memory.updated_at))

        # 分页
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        return query.all()

    def search_by_text(
        self,
        query_text: str,
        limit: int = 10
    ) -> List[Memory]:
        """
        文本搜索

        Args:
            query_text: 搜索文本
            limit: 限制数量

        Returns:
            List[Memory]: 匹配的记忆列表
        """
        return self.session.query(Memory).filter(
            or_(
                Memory.content.ilike(f"%{query_text}%"),
                Memory.summary.ilike(f"%{query_text}%")
            )
        ).limit(limit).all()

    def search_by_entities(
        self,
        entities: List[str],
        limit: int = 10
    ) -> List[Memory]:
        """
        根据实体搜索

        Args:
            entities: 实体列表
            limit: 限制数量

        Returns:
            List[Memory]: 匹配的记忆列表
        """
        query = self.session.query(Memory)

        conditions = []
        for entity in entities:
            # SQLAlchemy JSON contains expects a JSON string or appropriate construct,
            # but simpler approach for sqlite JSON is using func.json_extract or similar,
            # or just cast. We can use a custom filter or just func.json_each.
            # In SQLite, json_each is a table-valued function. But wait, SQLAlchemy JSON contains works on Postgres.
            # For SQLite compatibility, we'll fetch and filter in python if needed, or use like.
            # Actually, `like` is easier for SQLite JSON array string representation.
            conditions.append(Memory.entities.like(f'%"{entity}"%'))

        if conditions:
            query = query.filter(or_(*conditions))

        return query.limit(limit).all()

    def search_by_topics(
        self,
        topics: List[str],
        limit: int = 10
    ) -> List[Memory]:
        """
        根据主题搜索

        Args:
            topics: 主题列表
            limit: 限制数量

        Returns:
            List[Memory]: 匹配的记忆列表
        """
        query = self.session.query(Memory)

        conditions = []
        for topic in topics:
            conditions.append(Memory.topics.like(f'%"{topic}"%'))

        if conditions:
            query = query.filter(or_(*conditions))

        return query.limit(limit).all()

    def search_by_vector(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7,
        exclude_ids: Optional[List[int]] = None
    ) -> List[Tuple[Memory, float]]:
        """
        向量搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回前K个结果
            threshold: 相似度阈值
            exclude_ids: 要排除的记忆ID

        Returns:
            List[Tuple[Memory, float]]: (记忆, 相似度分数) 列表
        """
        return self.vector_store.search_similar_memories(
            query_embedding=query_embedding,
            session=self.session,
            top_k=top_k,
            threshold=threshold,
            exclude_ids=exclude_ids
        )

    def update(
        self,
        memory_id: int,
        **kwargs
    ) -> Optional[Memory]:
        """
        更新记忆

        Args:
            memory_id: 记忆ID
            **kwargs: 要更新的字段

        Returns:
            Optional[Memory]: 更新后的记忆对象或None
        """
        memory = self.get_by_id(memory_id)
        if not memory:
            return None

        # 更新字段
        for key, value in kwargs.items():
            if hasattr(memory, key):
                if key == "embedding" and value is not None:
                    # embedding需要特殊处理
                    value = VectorStore.embedding_to_json(value)
                setattr(memory, key, value)

        memory.updated_at = datetime.now()
        self.commit()

        self.logger.info(f"更新记忆: ID={memory_id}")
        return memory

    def delete(self, memory_id: int) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否成功删除
        """
        memory = self.get_by_id(memory_id)
        if not memory:
            return False

        super().delete(memory)
        self.commit()

        self.logger.info(f"删除记忆: ID={memory_id}")
        return True

    def count(self) -> int:
        """
        获取记忆总数

        Returns:
            int: 记忆数量
        """
        return self.session.query(Memory).count()


class FileRepository(BaseRepository):
    """文件仓储类"""

    def create(
        self,
        filename: str,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
        file_type: Optional[str] = None,
        content: Optional[str] = None
    ) -> FileUpload:
        """
        创建文件上传记录

        Args:
            filename: 文件名
            file_path: 文件路径
            file_size: 文件大小
            file_type: 文件类型
            content: 文件内容

        Returns:
            FileUpload: 文件对象
        """
        file_upload = FileUpload(
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_type,
            content=content,
            status="pending"
        )

        self.add(file_upload)
        self.commit()

        self.logger.info(f"创建文件上传记录: ID={file_upload.id}, 文件名={filename}")
        return file_upload

    def get_by_id(self, file_id: int) -> Optional[FileUpload]:
        """根据ID获取文件"""
        return self.session.query(FileUpload).filter(FileUpload.id == file_id).first()

    def get_by_status(self, status: str) -> List[FileUpload]:
        """根据状态获取文件列表"""
        return self.session.query(FileUpload).filter(FileUpload.status == status).all()

    def update_status(
        self,
        file_id: int,
        status: str,
        error_message: Optional[str] = None,
        memory_id: Optional[int] = None
    ) -> Optional[FileUpload]:
        """
        更新文件处理状态

        Args:
            file_id: 文件ID
            status: 新状态
            error_message: 错误信息
            memory_id: 关联的记忆ID

        Returns:
            Optional[FileUpload]: 更新后的文件对象
        """
        file_upload = self.get_by_id(file_id)
        if not file_upload:
            return None

        file_upload.status = status
        if error_message:
            file_upload.error_message = error_message
        if memory_id:
            file_upload.memory_id = memory_id
        if status == "completed":
            file_upload.processed_at = datetime.now()

        self.commit()

        return file_upload


class RelationRepository(BaseRepository):
    """记忆关系仓储类"""

    def create(
        self,
        memory_id: int,
        related_memory_id: int,
        relation_type: str = "similar",
        similarity_score: Optional[float] = None,
        created_by: str = "consolidate_agent",
        notes: Optional[str] = None
    ) -> MemoryRelation:
        """
        创建记忆关系

        Args:
            memory_id: 记忆ID
            related_memory_id: 关联的记忆ID
            relation_type: 关系类型
            similarity_score: 相似度分数
            created_by: 创建者
            notes: 备注

        Returns:
            MemoryRelation: 关系对象
        """
        relation = MemoryRelation(
            memory_id=memory_id,
            related_memory_id=related_memory_id,
            relation_type=relation_type,
            similarity_score=similarity_score,
            created_by=created_by,
            notes=notes
        )

        self.add(relation)
        self.commit()

        return relation

    def get_memory_relations(
        self,
        memory_id: int,
        relation_type: Optional[str] = None
    ) -> List[MemoryRelation]:
        """
        获取记忆的关系

        Args:
            memory_id: 记忆ID
            relation_type: 关系类型（可选）

        Returns:
            List[MemoryRelation]: 关系列表
        """
        query = self.session.query(MemoryRelation).filter(
            or_(
                MemoryRelation.memory_id == memory_id,
                MemoryRelation.related_memory_id == memory_id
            )
        )

        if relation_type:
            query = query.filter(MemoryRelation.relation_type == relation_type)

        return query.all()


    def get_relation(
        self,
        memory_id: int,
        related_memory_id: int
    ) -> Optional[MemoryRelation]:
        """
        获取关系
        """
        return self.session.query(MemoryRelation).filter(
            ((MemoryRelation.memory_id == memory_id) & (MemoryRelation.related_memory_id == related_memory_id)) |
            ((MemoryRelation.memory_id == related_memory_id) & (MemoryRelation.related_memory_id == memory_id))
        ).first()

    def delete_relation(self, relation_id: int) -> bool:
        """删除关系"""
        relation = self.session.query(MemoryRelation).filter(
            MemoryRelation.id == relation_id
        ).first()

        if relation:
            super().delete(relation)
            self.commit()
            return True
        return False
