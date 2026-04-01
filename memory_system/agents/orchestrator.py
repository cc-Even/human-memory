import logging
from typing import Dict, Any, Optional

from memory_system.agents.base_agent import BaseAgent
from memory_system.agents.ingest_agent import IngestAgent
from memory_system.agents.consolidate_agent import ConsolidateAgent
from memory_system.agents.query_agent import QueryAgent
from memory_system.agents.scheduler import ConsolidationScheduler, create_scheduler, get_scheduler
from memory_system.llm import LLMProvider, create_llm_provider
from memory_system.storage.database import DatabaseManager
from memory_system.config.settings import get_settings

logger = logging.getLogger(__name__)


class MemoryOrchestrator:
    """
    Memory Orchestrator (RootAgent)

    Acts as the "brain" or central dispatcher. It receives instructions,
    coordinates sub-agents (Ingest, Consolidate, Query), and manages the workflow.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        db_manager: Optional[DatabaseManager] = None,
        embedding_provider: Optional[LLMProvider] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the MemoryOrchestrator.

        Args:
            llm_provider: LLM provider for text processing (optional, will create from settings if not provided).
            db_manager: Database manager (optional, will create from settings if not provided).
            embedding_provider: LLM provider for embeddings (optional, will use llm_provider if not provided).
            config: Configuration dictionary (optional, will load from settings if not provided).
        """
        self.logger = logger
        self.settings = get_settings()

        # Use provided instances or create from settings
        if llm_provider is None:
            llm_provider = self._create_chat_provider()
        if db_manager is None:
            db_manager = self._create_db_manager()
        if embedding_provider is None:
            embedding_provider = llm_provider

        self.llm_provider = llm_provider
        self.db_manager = db_manager
        self.embedding_provider = embedding_provider
        self.config = config or self._load_config_from_settings()

        # Initialize sub-agents
        self.ingest_agent = IngestAgent(
            llm_provider=self.llm_provider,
            db_manager=self.db_manager,
            embedding_provider=self.embedding_provider,
            config=self.config.get("ingest", {})
        )
        self.consolidate_agent = ConsolidateAgent(
            llm_provider=self.llm_provider,
            db_manager=self.db_manager,
            embedding_provider=self.embedding_provider,
            config=self.config.get("consolidate", {})
        )
        self.query_agent = QueryAgent(
            llm_provider=self.llm_provider,
            db_manager=self.db_manager,
            config=self.config.get("query", {})
        )

        # Initialize scheduler
        self._scheduler: Optional[ConsolidationScheduler] = None
        self._init_scheduler()

        self.logger.info("Memory Orchestrator initialized with sub-agents.")

    def _init_scheduler(self):
        """初始化调度器"""
        consolidate_config = self.config.get("consolidate", {})
        enabled = consolidate_config.get("auto_consolidate_enabled", False)
        interval_hours = consolidate_config.get("consolidation_interval_hours", 24)

        if enabled:
            self._scheduler = create_scheduler(
                consolidate_func=lambda: self.consolidate_agent.consolidate(
                    time_window_hours=interval_hours
                ),
                interval_hours=interval_hours,
                enabled=enabled
            )
            self.logger.info(f"调度器已配置: interval={interval_hours}h, enabled={enabled}")

    def _create_chat_provider(self) -> LLMProvider:
        """从设置创建聊天提供商"""
        llm_settings = self.settings.llm
        provider_name = llm_settings.default_llm_provider

        if provider_name == "dashscope":
            api_key = llm_settings.dashscope_api_key
            kwargs = {
                "chat_model": llm_settings.dashscope_chat_model,
            }
        elif provider_name == "gemini":
            api_key = llm_settings.gemini_api_key
            kwargs = {
                "model": llm_settings.gemini_chat_model,
            }
        elif provider_name == "openai":
            api_key = llm_settings.openai_api_key
            kwargs = {
                "base_url": llm_settings.openai_base_url,
                "chat_model": llm_settings.openai_chat_model,
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

        return create_llm_provider(provider_name, api_key, **kwargs)

    def _create_embedding_provider(self) -> LLMProvider:
        """从设置创建embedding提供商"""
        llm_settings = self.settings.llm
        provider_name = llm_settings.embedding_provider

        if provider_name == "dashscope":
            api_key = llm_settings.dashscope_api_key
            kwargs = {
                "embedding_model": llm_settings.dashscope_embedding_model,
            }
        elif provider_name == "gemini":
            api_key = llm_settings.gemini_api_key
            kwargs = {
                "model": llm_settings.gemini_embedding_model,
            }
        elif provider_name == "openai":
            api_key = llm_settings.openai_api_key
            kwargs = {
                "base_url": llm_settings.openai_base_url,
                "embedding_model": llm_settings.openai_embedding_model,
            }
        else:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")

        return create_llm_provider(provider_name, api_key, **kwargs)

    def _create_db_manager(self) -> DatabaseManager:
        """从设置创建数据库管理器"""
        from memory_system.storage.database import DatabaseManager
        db_path = self.settings.database.database_path
        return DatabaseManager(db_path)

    def _load_config_from_settings(self) -> Dict[str, Any]:
        """从设置加载配置"""
        return {
            "query": {
                "top_k": self.settings.query.top_k,
                "relevance_threshold": self.settings.query.relevance_threshold,
                "max_context_length": self.settings.query.max_context_length,
                "enable_vector_search": self.settings.query.enable_vector_search,
                "enable_conversation": self.settings.query.enable_conversation,
                "temperature": self.settings.query.temperature,
                "max_tokens": self.settings.query.max_tokens,
            },
            "consolidate": {
                "similarity_threshold": self.settings.consolidate.similarity_threshold,
                "top_k": self.settings.consolidate.top_k,
                "max_merge_count": self.settings.consolidate.max_merge_count,
                "consolidation_interval_hours": self.settings.consolidate.consolidation_interval_hours,
                "auto_consolidate_enabled": self.settings.consolidate.auto_consolidate_enabled,
            }
        }

    def process_input(self, text: str) -> str:
        """
        Process user input. Determines if the input is a query or new information to ingest.
        For simplicity, if it sounds like a query, we route to QueryAgent, otherwise IngestAgent.
        """
        self.logger.info(f"Orchestrator received input: {text}")
        
        # A simple heuristic or LLM call could be used here to classify the intent.
        # We will use the LLM to classify the intent.
        intent = self._classify_intent(text)
        
        if intent == "query":
            self.logger.info("Routing to QueryAgent")
            result = self.query_agent.process(text)
            if result.get("success"):
                return result.get("answer", "I couldn't generate an answer.")
            return result.get("error", "Error processing query.")
            
        elif intent == "ingest":
            self.logger.info("Routing to IngestAgent")
            try:
                memory = self.ingest_agent.ingest_text(text)
                return f"已保存记忆！(ID: {memory.id})"
            except Exception as e:
                self.logger.error(f"Ingest error: {e}")
                return f"保存记忆失败: {e}"
                
        elif intent == "consolidate":
            self.logger.info("Routing to ConsolidateAgent")
            return self.trigger_consolidation()
            
        else:
            return "I did not understand the intent."

    def _classify_intent(self, text: str) -> str:
        """Classify the user input intent: query, ingest, or consolidate."""
        prompt = f"""Classify the following user input into one of these categories:
1. 'query': The user is asking a question or trying to retrieve information.
2. 'ingest': The user is providing new information to be remembered or saved.
3. 'consolidate': The user is asking to organize, summarize, or consolidate memories.

User input: "{text}"

Output only the category name (query, ingest, or consolidate) in lowercase without quotes.
"""
        from memory_system.llm.base import ChatMessage
        messages = [ChatMessage(role="user", content=prompt)]
        response = self.llm_provider.chat(messages, temperature=0.1)
        intent = response.content.strip().lower()
        if "query" in intent:
            return "query"
        elif "consolidate" in intent:
            return "consolidate"
        else:
            # Default to ingest if we can't tell
            return "ingest"

    def ingest_file(self, file_path: str) -> str:
        """Ingest a file through the IngestAgent."""
        self.logger.info(f"Orchestrator ingesting file: {file_path}")
        try:
            memory = self.ingest_agent.ingest_file(file_path)
            return f"已摄取文件内容！(Memory ID: {memory.id})"
        except Exception as e:
            self.logger.error(f"File ingest error: {e}")
            return f"文件摄取失败: {e}"

    def trigger_consolidation(self, time_window_hours: int = 24) -> str:
        """Trigger the ConsolidateAgent to consolidate recent memories."""
        self.logger.info("Orchestrator triggering consolidation")
        try:
            results = self.consolidate_agent.consolidate(time_window_hours=time_window_hours)
            return f"已整合相关记忆！处理了 {results.get('processed', 0)} 条记录，创建了 {results.get('relations_created', 0)} 个关联。"
        except Exception as e:
            self.logger.error(f"Consolidation error: {e}")
            return f"记忆整合失败: {e}"

    def start_scheduler(self) -> str:
        """启动自动整合调度器"""
        if self._scheduler is None:
            return "调度器未配置，请设置 AUTO_CONSOLIDATE_ENABLED=true"
        if self._scheduler.is_running:
            return "调度器已在运行"
        self._scheduler.start()
        return "调度器已启动"

    def stop_scheduler(self) -> str:
        """停止自动整合调度器"""
        if self._scheduler is None:
            return "调度器未配置"
        if not self._scheduler.is_running:
            return "调度器未在运行"
        self._scheduler.stop()
        return "调度器已停止"

    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        if self._scheduler is None:
            return {"enabled": False, "running": False}
        return {
            "enabled": True,
            "running": self._scheduler.is_running,
            "last_run": self._scheduler.last_run_time.isoformat() if self._scheduler.last_run_time else None,
            "interval_hours": self._scheduler.interval_hours
        }

    def shutdown(self):
        """关闭orchestrator及其相关资源"""
        self.logger.info("关闭 MemoryOrchestrator...")
        if self._scheduler and self._scheduler.is_running:
            self._scheduler.stop()
        self.logger.info("MemoryOrchestrator 已关闭")
