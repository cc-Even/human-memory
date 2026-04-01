"""
日志系统配置模块
提供统一的日志格式和分级管理
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


class Logger:
    """日志管理器类"""

    _loggers: dict = {}

    @staticmethod
    def get_logger(
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        max_size: int = 10,
        backup_count: int = 5,
    ) -> logging.Logger:
        """
        获取或创建日志记录器

        Args:
            name: 日志记录器名称
            level: 日志级别
            log_file: 日志文件路径（可选）
            max_size: 日志文件最大大小(MB)
            backup_count: 保留的备份文件数量

        Returns:
            logging.Logger: 日志记录器实例
        """
        if name in Logger._loggers:
            return Logger._loggers[name]

        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # 防止重复添加handler
        if logger.handlers:
            return logger

        # 日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件处理器（如果指定了日志文件）
        if log_file:
            # 确保日志目录存在
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # 按大小轮转的文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_size * 1024 * 1024,  # 转换为字节
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        Logger._loggers[name] = logger
        return logger


# 预定义的日志记录器实例
def get_module_logger(module_name: str) -> logging.Logger:
    """
    获取模块专用日志记录器

    Args:
        module_name: 模块名称（如 'agents.ingest'）

    Returns:
        logging.Logger: 日志记录器实例
    """
    # 从环境变量读取配置
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "./logs/app.log")

    return Logger.get_logger(module_name, log_level, log_file)


# 快捷获取各个模块的日志记录器
def get_agents_logger(agent_name: str) -> logging.Logger:
    """获取智能体模块的日志记录器"""
    return get_module_logger(f"agents.{agent_name}")


def get_agent_logger() -> logging.Logger:
    """获取通用Agent日志记录器（向后兼容）"""
    return get_module_logger("agents")


def get_storage_logger() -> logging.Logger:
    """获取存储模块的日志记录器"""
    return get_module_logger("storage")


def get_llm_logger() -> logging.Logger:
    """获取LLM模块的日志记录器"""
    return get_module_logger("llm")


def get_ui_logger() -> logging.Logger:
    """获取UI模块的日志记录器"""
    return get_module_logger("ui")
