"""
记忆系统调度器
提供自动化的定时任务功能
"""

import logging
import threading
import time
from typing import Optional, Callable
from datetime import datetime

from memory_system.utils.logger import get_agent_logger

logger = get_agent_logger()


class ConsolidationScheduler:
    """
    记忆整合调度器

    负责定期执行记忆整合任务
    """

    def __init__(
        self,
        consolidate_func: Callable,
        interval_hours: int = 24,
        enabled: bool = False
    ):
        """
        初始化调度器

        Args:
            consolidate_func: 整合函数，会被定期调用
            interval_hours: 执行间隔（小时）
            enabled: 是否启用
        """
        self.consolidate_func = consolidate_func
        self.interval_hours = interval_hours
        self.enabled = enabled
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_run: Optional[datetime] = None
        self._last_result: Optional[dict] = None

    def start(self):
        """启动调度器"""
        if not self.enabled:
            logger.info("调度器未启用")
            return

        if self._thread is not None and self._thread.is_alive():
            logger.warning("调度器已在运行中")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"调度器已启动，间隔={self.interval_hours}小时")

    def stop(self):
        """停止调度器"""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None
        logger.info("调度器已停止")

    def _run(self):
        """调度器主循环"""
        # 首次执行前等待一个interval
        interval_seconds = self.interval_hours * 3600

        while not self._stop_event.wait(interval_seconds):
            if self._stop_event.is_set():
                break
            self._execute_consolidation()

        # 确保最后一次执行
        self._execute_consolidation()

    def _execute_consolidation(self):
        """执行整合任务"""
        logger.info("开始执行定时整合任务...")
        try:
            self._last_result = self.consolidate_func()
            self._last_run = datetime.now()
            logger.info(f"定时整合任务完成: {self._last_result}")
        except Exception as e:
            logger.error(f"定时整合任务失败: {e}")

    def run_now(self) -> Optional[dict]:
        """
        立即执行一次整合

        Returns:
            dict: 整合结果
        """
        logger.info("手动触发整合任务...")
        self._execute_consolidation()
        return self._last_result

    @property
    def is_running(self) -> bool:
        """检查调度器是否在运行"""
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_run_time(self) -> Optional[datetime]:
        """获取上次执行时间"""
        return self._last_run

    @property
    def last_result(self) -> Optional[dict]:
        """获取上次执行结果"""
        return self._last_result


# 全局调度器实例
_scheduler: Optional[ConsolidationScheduler] = None


def get_scheduler() -> Optional[ConsolidationScheduler]:
    """获取全局调度器实例"""
    global _scheduler
    return _scheduler


def create_scheduler(
    consolidate_func: Callable,
    interval_hours: int = 24,
    enabled: bool = False
) -> ConsolidationScheduler:
    """
    创建调度器实例

    Args:
        consolidate_func: 整合函数
        interval_hours: 执行间隔
        enabled: 是否启用

    Returns:
        ConsolidationScheduler: 调度器实例
    """
    global _scheduler
    _scheduler = ConsolidationScheduler(
        consolidate_func=consolidate_func,
        interval_hours=interval_hours,
        enabled=enabled
    )
    return _scheduler
