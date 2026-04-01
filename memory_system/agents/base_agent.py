"""
Agent基类
为所有Agent提供统一的接口和基础功能
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from memory_system.llm import ChatMessage, LLMProvider
from memory_system.utils.logger import get_agent_logger


class BaseAgent(ABC):
    """
    Agent基类

    所有具体的Agent（IngestAgent, ConsolidateAgent, QueryAgent）都需要继承这个类
    """

    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化Agent

        Args:
            name: Agent名称
            llm_provider: LLM提供商实例
            system_prompt: 系统提示词
            config: 其他配置参数
        """
        self.name = name
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt
        self.config = config or {}
        self.logger = get_agent_logger()

        self.logger.info(f"{self.name} 初始化完成")

    @abstractmethod
    def process(self, *args, **kwargs):
        """
        处理任务的抽象方法

        各个具体的Agent需要实现这个方法
        """
        pass

    def chat(
        self,
        user_message: str,
        history: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        与LLM进行对话

        Args:
            user_message: 用户消息
            history: 历史消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            str: LLM响应
        """
        messages = []

        # 添加系统提示词
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))

        # 添加历史消息
        if history:
            messages.extend(history)

        # 添加当前用户消息
        messages.append(ChatMessage(role="user", content=user_message))

        # 调用LLM
        try:
            response = self.llm_provider.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            self.logger.debug(f"{self.name} 调用LLM成功")
            return response.content

        except Exception as e:
            self.logger.error(f"{self.name} 调用LLM失败: {e}")
            raise

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键
            default: 默认值

        Returns:
            Any: 配置值
        """
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any):
        """
        设置配置项

        Args:
            key: 配置键
            value: 配置值
        """
        self.config[key] = value

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}')>"
