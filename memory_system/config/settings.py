"""
配置管理模块
从环境变量加载和管理应用配置
"""

import os
from typing import Optional
from pathlib import Path
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """LLM配置"""
    model_config = ConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # DashScope配置
    dashscope_api_key: Optional[str] = Field(None, alias="DASHSCOPE_API_KEY")
    dashscope_embedding_model: str = Field(
        "text-embedding-v2",
        alias="DASHSCOPE_EMBEDDING_MODEL"
    )
    dashscope_chat_model: str = Field(
        "qwen-plus",
        alias="DASHSCOPE_CHAT_MODEL"
    )

    # Gemini配置
    gemini_api_key: Optional[str] = Field(None, alias="GEMINI_API_KEY")
    gemini_chat_model: str = Field(
        "gemini-2.0-flash",
        alias="GEMINI_CHAT_MODEL"
    )
    gemini_embedding_model: str = Field(
        "gemini-embedding-001",
        alias="GEMINI_EMBEDDING_MODEL"
    )

    # OpenAI配置
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(None, alias="OPENAI_BASE_URL")
    openai_chat_model: str = Field(
        "gpt-3.5-turbo",
        alias="OPENAI_CHAT_MODEL"
    )
    openai_embedding_model: str = Field(
        "text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL"
    )

    # 默认提供商
    default_llm_provider: str = Field(
        "gemini",
        alias="DEFAULT_LLM_PROVIDER"
    )

    @field_validator("default_llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """验证提供商名称"""
        v = v.lower()
        if v not in ["dashscope", "gemini", "openai"]:
            raise ValueError(
                f"不支持的LLM提供商: {v}。"
                f"支持的提供商: dashscope, gemini, openai"
            )
        return v

    @property
    def embedding_provider(self) -> str:
        """获取embedding提供商"""
        if self.default_llm_provider == "openai":
            return "openai"
        elif self.default_llm_provider == "gemini":
            return "gemini"
        return "dashscope"


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    model_config = ConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    database_path: str = Field(
        "./memory.db",
        alias="DATABASE_PATH"
    )

    # 向量检索配置
    vector_search_top_k: int = Field(
        5,
        alias="VECTOR_SEARCH_TOP_K"
    )
    similarity_threshold: float = Field(
        0.7,
        alias="SIMILARITY_THRESHOLD"
    )

    @field_validator("vector_search_top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("VECTOR_SEARCH_TOP_K必须在1-100之间")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("SIMILARITY_THRESHOLD必须在0-1之间")
        return v


class QueryAgentConfig(BaseSettings):
    """QueryAgent配置"""
    model_config = ConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # 返回的记忆数量
    top_k: int = Field(5, alias="QUERY_TOP_K")
    # 相关度阈值
    relevance_threshold: float = Field(0.3, alias="QUERY_RELEVANCE_THRESHOLD")
    # 最大上下文长度
    max_context_length: int = Field(4000, alias="QUERY_MAX_CONTEXT_LENGTH")
    # 是否启用向量搜索
    enable_vector_search: bool = Field(True, alias="QUERY_ENABLE_VECTOR_SEARCH")
    # 是否支持多轮对话
    enable_conversation: bool = Field(True, alias="QUERY_ENABLE_CONVERSATION")
    # 温度参数
    temperature: float = Field(0.7, alias="QUERY_TEMPERATURE")
    # 最大token数
    max_tokens: int = Field(1000, alias="QUERY_MAX_TOKENS")


class ConsolidateAgentConfig(BaseSettings):
    """ConsolidateAgent配置"""
    model_config = ConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # 相似度阈值
    similarity_threshold: float = Field(0.75, alias="CONSILIDATE_SIMILARITY_THRESHOLD")
    # 返回的记忆数量
    top_k: int = Field(5, alias="CONSILIDATE_TOP_K")
    # 最大合并数量
    max_merge_count: int = Field(3, alias="CONSILIDATE_MAX_MERGE_COUNT")
    # 自动整合时间窗口（小时）
    consolidation_interval_hours: int = Field(24, alias="CONSILIDATION_INTERVAL_HOURS")
    # 是否启用自动整合
    auto_consolidate_enabled: bool = Field(False, alias="AUTO_CONSOLIDATE_ENABLED")


class LogConfig(BaseSettings):
    """日志配置"""
    model_config = ConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    log_level: str = Field(
        "INFO",
        alias="LOG_LEVEL"
    )
    log_file: str = Field(
        "./logs/app.log",
        alias="LOG_FILE"
    )
    log_max_size: int = Field(
        10,
        alias="LOG_MAX_SIZE"
    )
    log_backup_count: int = Field(
        5,
        alias="LOG_BACKUP_COUNT"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v = v.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(
                f"不支持的日志级别: {v}。"
                f"支持的级别: {', '.join(valid_levels)}"
            )
        return v


class Settings(BaseSettings):
    """
    全局配置类
    集成所有子配置模块
    """

    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    query: QueryAgentConfig = Field(default_factory=QueryAgentConfig)
    consolidate: ConsolidateAgentConfig = Field(default_factory=ConsolidateAgentConfig)

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @classmethod
    def load(cls, env_file: Optional[str] = None) -> "Settings":
        """
        加载配置

        Args:
            env_file: 环境变量文件路径（可选）

        Returns:
            Settings: 配置实例
        """
        if env_file:
            if not Path(env_file).exists():
                raise FileNotFoundError(f"配置文件不存在: {env_file}")
            return Settings(_env_file=env_file)
        return Settings()

    def validate_api_keys(self) -> dict:
        """
        验证API密钥配置

        Returns:
            dict: 包含可用提供商的字典

        Raises:
            ValueError: 缺少必需的API密钥
        """
        available_providers = {}

        # 检查DashScope
        if self.llm.dashscope_api_key:
            available_providers["dashscope"] = True
        else:
            available_providers["dashscope"] = False

        # 检查Gemini
        if self.llm.gemini_api_key:
            available_providers["gemini"] = True
        else:
            available_providers["gemini"] = False

        # 检查OpenAI
        if self.llm.openai_api_key:
            available_providers["openai"] = True
        else:
            available_providers["openai"] = False

        # 检查默认提供商是否可用
        default_provider = self.llm.default_llm_provider
        if not available_providers.get(default_provider, False):
            raise ValueError(
                f"默认LLM提供商 '{default_provider}' 的API密钥未配置。"
                f"请在.env文件中设置相应的API_KEY。"
            )

        return available_providers

    def ensure_required_configs(self):
        """
        确保必需的配置项已设置

        Raises:
            ValueError: 缺少必需配置
        """
        # 至少需要一个对话API密钥
        has_chat_api = bool(
            self.llm.dashscope_api_key or 
            self.llm.gemini_api_key or 
            self.llm.openai_api_key
        )

        if not has_chat_api:
            raise ValueError(
                "缺少对话API密钥。请在.env文件中配置 "
                "DASHSCOPE_API_KEY, GEMINI_API_KEY 或 OPENAI_API_KEY"
            )

        # 需要embedding API密钥（支持DashScope、OpenAI和Gemini）
        has_embedding_api = bool(
            self.llm.dashscope_api_key or 
            self.llm.openai_api_key or
            self.llm.gemini_api_key
        )

        if not has_embedding_api:
            raise ValueError(
                "缺少Embedding API密钥。系统需要支持Embedding的提供商。"
                "请在.env文件中配置 DASHSCOPE_API_KEY、OPENAI_API_KEY 或 GEMINI_API_KEY"
            )


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings(env_file: Optional[str] = None) -> Settings:
    """
    获取全局配置实例

    Args:
        env_file: 环境变量文件路径（可选）

    Returns:
        Settings: 配置实例
    """
    global _settings
    if _settings is None:
        _settings = Settings.load(env_file)
    return _settings


def reload_settings(env_file: Optional[str] = None) -> Settings:
    """
    重新加载配置

    Args:
        env_file: 环境变量文件路径（可选）

    Returns:
        Settings: 新的配置实例
    """
    global _settings
    _settings = Settings.load(env_file)
    return _settings
