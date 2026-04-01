"""
数据库ORM模型定义
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, 
    JSON, Index, ForeignKey, CheckConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Memory(Base):
    """
    记忆模型
    存储用户摄入的信息及其处理结果
    """
    __tablename__ = "memories"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 内容字段
    content = Column(Text, nullable=False, comment="原始内容")
    summary = Column(Text, comment="内容摘要")
    
    # JSON字段
    entities = Column(JSON, comment="提取的实体列表")
    topics = Column(JSON, comment="主题标签列表")
    
    # 向量字段（存储为BLOB）
    embedding = Column(String, comment="向量化后的字符串表示")
    
    # 元数据
    source_type = Column(
        String(50),
        default="user_input",
        comment="来源类型：user_input, file_upload"
    )
    source_id = Column(Integer, nullable=True, comment="来源ID（如文件ID）")

    # 合并状态
    is_merged = Column(Integer, default=0, comment="是否已合并：0=否, 1=是")
    merged_into = Column(Integer, nullable=True, comment="合并后的记忆ID")

    # 重要性评分
    importance_score = Column(Float, default=0.5, comment="重要性评分 0-1")

    # 时间戳
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # 关系
    relations_from = relationship(
        "MemoryRelation",
        foreign_keys="MemoryRelation.memory_id",
        back_populates="memory",
        cascade="all, delete-orphan"
    )
    relations_to = relationship(
        "MemoryRelation",
        foreign_keys="MemoryRelation.related_memory_id",
        back_populates="related_memory",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Memory(id={self.id}, content='{self.content[:50]}...', created_at={self.created_at})>"

    def get_entities_list(self) -> List[str]:
        """获取实体列表"""
        return self.entities if self.entities else []

    def get_topics_list(self) -> List[str]:
        """获取主题列表"""
        return self.topics if self.topics else []


class FileUpload(Base):
    """
    文件上传模型
    存储上传的文件信息
    """
    __tablename__ = "files"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 文件信息
    filename = Column(String(255), nullable=False, comment="文件名")
    file_path = Column(String(512), nullable=True, comment="文件存储路径")
    file_size = Column(Integer, comment="文件大小（字节）")
    file_type = Column(String(100), comment="文件类型/MIME类型")
    
    # 内容
    content = Column(Text, comment="提取的文本内容")
    
    # 处理状态
    status = Column(
        String(50),
        default="pending",
        comment="处理状态：pending, processing, completed, failed"
    )
    error_message = Column(Text, comment="错误信息")
    
    # 关联的记忆ID
    memory_id = Column(Integer, ForeignKey("memories.id"), nullable=True)
    memory = relationship("Memory", backref="file_source")
    
    # 时间戳
    uploaded_at = Column(DateTime, default=func.now(), nullable=False)
    processed_at = Column(DateTime, comment="处理完成时间")

    def __repr__(self):
        return f"<FileUpload(id={self.id}, filename='{self.filename}', status='{self.status}')>"


class MemoryRelation(Base):
    """
    记忆关系模型
    存储记忆之间的关联信息
    """
    __tablename__ = "memory_relations"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 关联的记忆
    memory_id = Column(
        Integer,
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    related_memory_id = Column(
        Integer,
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # 关系属性
    relation_type = Column(
        String(50),
        default="similar",
        comment="关系类型：similar, duplicate, related, consolidated"
    )
    similarity_score = Column(
        Float,
        comment="相似度分数 (0-1)"
    )
    
    # 元数据
    created_by = Column(
        String(50),
        default="consolidate_agent",
        comment="创建者：consolidate_agent, user"
    )
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # 备注
    notes = Column(Text, comment="关系备注")

    # 关系
    memory = relationship("Memory", foreign_keys=[memory_id], back_populates="relations_from")
    related_memory = relationship("Memory", foreign_keys=[related_memory_id], back_populates="relations_to")

    # 确保不创建自关联
    __table_args__ = (
        CheckConstraint("memory_id != related_memory_id", name="check_not_self_relation"),
        Index("idx_similarity_score", "similarity_score"),
    )

    def __repr__(self):
        return (
            f"<MemoryRelation(id={self.id}, "
            f"memory_id={self.memory_id}, "
            f"related_memory_id={self.related_memory_id}, "
            f"score={self.similarity_score})>"
        )


# 导出所有模型
__all__ = ["Base", "Memory", "FileUpload", "MemoryRelation"]
