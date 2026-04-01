#!/usr/bin/env python3
"""
数据库初始化脚本
创建数据库表和初始数据
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from memory_system.storage.database import init_database, get_db_manager
from memory_system.config.settings import get_settings
from memory_system.utils.logger import get_storage_logger


def create_sample_data(db_manager):
    """
    创建示例数据

    Args:
        db_manager: 数据库管理器
    """
    from memory_system.storage.models import Memory, FileUpload
    from datetime import datetime

    with db_manager.get_session() as session:
        mem_repo = db_manager.get_memory_repository(session)
        file_repo = db_manager.get_file_repository(session)

        # 创建一些示例记忆
        sample_memories = [
            {
                "content": "今天学习了Python的异步编程，asyncio库很强大，可以高效处理并发任务。",
                "summary": "学习Python异步编程",
                "entities": ["Python", "asyncio"],
                "topics": ["编程", "学习"]
            },
            {
                "content": "参加了技术分享会，讨论了微服务架构的设计原则和实践经验。",
                "summary": "参加微服务技术分享会",
                "entities": ["微服务", "架构"],
                "topics": ["技术", "分享"]
            },
            {
                "content": "阅读了《代码整洁之道》这本书，对编写高质量代码有了新的理解。",
                "summary": "阅读代码整洁之道",
                "entities": ["代码整洁之道", "书籍"],
                "topics": ["阅读", "编程"]
            }
        ]

        for i, mem_data in enumerate(sample_memories):
            memory = mem_repo.create(**mem_data)
            print(f"✓ 创建示例记忆 {i+1}: {memory.summary}")

        # 创建示例文件记录
        sample_files = [
            {
                "filename": "notes.txt",
                "file_size": 1024,
                "file_type": "text/plain",
                "content": "这是从文件中提取的示例文本内容。"
            },
            {
                "filename": "meeting_minutes.md",
                "file_size": 2048,
                "file_type": "text/markdown",
                "content": "会议记录：讨论了项目进度和下一步计划。"
            }
        ]

        for i, file_data in enumerate(sample_files):
            file_record = file_repo.create(**file_data)
            # 更新为已完成状态
            file_repo.update_status(file_record.id, "completed")
            print(f"✓ 创建示例文件记录 {i+1}: {file_record.filename}")


def print_db_stats(db_manager):
    """
    打印数据库统计信息

    Args:
        db_manager: 数据库管理器
    """
    stats = db_manager.get_db_stats()

    print("\n" + "="*50)
    print("数据库统计信息")
    print("="*50)
    print(f"数据库路径: {stats['db_path']}")
    print(f"数据库大小: {stats['db_size_mb']} MB")
    print(f"记忆数量: {stats['memory_count']}")
    print(f"文件数量: {stats['file_count']}")
    print(f"关系数量: {stats['relation_count']}")
    print("="*50)


def main():
    """主函数"""
    logger = get_storage_logger()

    # 获取配置
    settings = get_settings()
    db_path = settings.database.database_path

    print("\n" + "="*50)
    print("Memory System - 数据库初始化")
    print("="*50)

    # 检查数据库是否已存在
    if os.path.exists(db_path):
        response = input(f"\n数据库文件 '{db_path}' 已存在。是否继续？[y/N]: ").strip().lower()
        if response != 'y':
            print("操作已取消")
            return

        response = input("是否重置数据库（删除所有数据）？[y/N]: ").strip().lower()
        if response == 'y':
            print("\n⚠️  正在重置数据库...")
            db_manager = get_db_manager(db_path)
            db_manager.reset_db()
        else:
            print("\n继续初始化数据库表...")
            db_manager = init_database(db_path)
    else:
        print(f"\n创建新数据库: {db_path}")
        db_manager = init_database(db_path)

    # 询问是否创建示例数据
    response = input("\n是否创建示例数据？[Y/n]: ").strip().lower()
    if response != 'n':
        print("\n正在创建示例数据...")
        create_sample_data(db_manager)
    else:
        print("\n跳过创建示例数据")

    # 打印统计信息
    print_db_stats(db_manager)

    print("\n✅ 数据库初始化完成！\n")


if __name__ == "__main__":
    main()
