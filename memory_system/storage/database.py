"""
数据库连接和初始化管理
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from memory_system.storage.models import Base
from memory_system.storage.repository import (
    MemoryRepository,
    FileRepository,
    RelationRepository
)
from memory_system.utils.logger import get_storage_logger


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, db_path: str = "memory.db"):
        """
        初始化数据库管理器

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.logger = get_storage_logger()

        # 确保数据库目录存在
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # 创建数据库引擎
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,  # 设置为True可以看到SQL日志
            pool_pre_ping=True  # 连接健康检查
        )

        # 创建会话工厂
        self.SessionLocal = sessionmaker(expire_on_commit=False, 
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        self.logger.info(f"数据库管理器初始化完成: {db_path}")

    def init_db(self):
        """
        初始化数据库，创建所有表
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("数据库表创建成功")
            return True
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            raise

    def drop_all(self):
        """
        删除所有表（谨慎使用！）
        """
        try:
            Base.metadata.drop_all(bind=self.engine)
            self.logger.warning("所有数据库表已删除")
            return True
        except Exception as e:
            self.logger.error(f"删除表失败: {e}")
            raise

    def reset_db(self):
        """
        重置数据库（删除并重新创建）
        """
        self.drop_all()
        self.init_db()
        self.logger.info("数据库已重置")

    @contextmanager
    def get_session(self) -> Session:
        """
        获取数据库会话的上下文管理器

        Yields:
            Session: 数据库会话
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"数据库操作失败，已回滚: {e}")
            raise
        finally:
            session.close()

    def get_memory_repository(self, session: Session) -> MemoryRepository:
        """
        获取记忆仓储

        Args:
            session: 数据库会话

        Returns:
            MemoryRepository: 记忆仓储实例
        """
        return MemoryRepository(session, self.db_path)

    def get_file_repository(self, session: Session) -> FileRepository:
        """
        获取文件仓储

        Args:
            session: 数据库会话

        Returns:
            FileRepository: 文件仓储实例
        """
        return FileRepository(session)

    def get_relation_repository(self, session: Session) -> RelationRepository:
        """
        获取关系仓储

        Args:
            session: 数据库会话

        Returns:
            RelationRepository: 关系仓储实例
        """
        return RelationRepository(session)

    def get_db_stats(self) -> dict:
        """
        获取数据库统计信息

        Returns:
            dict: 统计信息
        """
        with self.get_session() as session:
            from memory_system.storage.models import Memory, FileUpload, MemoryRelation

            stats = {
                "memory_count": session.query(Memory).count(),
                "file_count": session.query(FileUpload).count(),
                "relation_count": session.query(MemoryRelation).count(),
                "db_path": self.db_path,
                "db_size_mb": round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if os.path.exists(self.db_path) else 0
            }

        return stats


# 全局数据库管理器实例
_db_manager = None


def get_db_manager(db_path: str = "memory.db") -> DatabaseManager:
    """
    获取全局数据库管理器实例

    Args:
        db_path: 数据库文件路径

    Returns:
        DatabaseManager: 数据库管理器实例
    """
    global _db_manager

    if _db_manager is None or _db_manager.db_path != db_path:
        _db_manager = DatabaseManager(db_path)

    return _db_manager


def init_database(db_path: str = "memory.db") -> DatabaseManager:
    """
    初始化数据库（便捷函数）

    Args:
        db_path: 数据库文件路径

    Returns:
        DatabaseManager: 数据库管理器实例
    """
    db_manager = get_db_manager(db_path)
    db_manager.init_db()
    return db_manager
