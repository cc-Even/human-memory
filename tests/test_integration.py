import unittest
import os
import sys
import tempfile
import sqlite3
from unittest.mock import Mock, MagicMock, patch

# Mock modules
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()
sys.modules['dashscope'] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory_system.agents.orchestrator import MemoryOrchestrator
from memory_system.storage.database import DatabaseManager
from memory_system.storage.models import Base

class TestMemoryIntegration(unittest.TestCase):
    def setUp(self):
        # Create temp dir for database
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_memory.db")
        
        # Init DB Manager
        self.db_manager = DatabaseManager(self.db_path)
        self.db_manager.init_db()
        
        # Create mock LLM Provider
        self.mock_llm = MagicMock()
        self.mock_llm.supports_embedding.return_value = True
        self.mock_llm.get_embedding.return_value = [0.1] * 1536 # Mock embedding vector
        
        # Configure Orchestrator
        self.config = {
            "ingest_agent": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "extract_entities": True
            },
            "consolidate_agent": {
                "similarity_threshold": 0.8,
                "time_decay_factor": 0.1
            },
            "query_agent": {
                "top_k": 3,
                "relevance_threshold": 0.5,
                "enable_vector_search": False
            }
        }
        
        # Create Orchestrator with mock LLM
        self.orchestrator = MemoryOrchestrator(
            db_manager=self.db_manager,
            llm_provider=self.mock_llm,
            config=self.config
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_end_to_end_flow(self):
        # 1. Ingest Step
        input_text = "Even loves programming in Python."
        
        # Mocking extraction behavior for IngestAgent
        import json
        self.mock_llm.chat.return_value = MagicMock(content=json.dumps({
            "summary": "Even's programming preferences.",
            "entities": ["Even", "Python"],
            "topics": ["Programming", "Python"],
            "importance": 0.8
        }))
        
        # Mock _classify_intent
        with patch.object(self.orchestrator, '_classify_intent', return_value='ingest'):
            result_add = self.orchestrator.process_input(input_text)
            self.assertIn("已保存记忆！", result_add)
            # Extra memory ID from "已保存记忆！(ID: 1)"
            memory_id = int(result_add.split("(ID: ")[1].split(")")[0])
        
        # 2. Query Step
        query = "What programming language does Even love?"
        
        # Mock behavior for QueryAgent
        self.mock_llm.chat.side_effect = [
            # extract_search_terms
            MagicMock(content=json.dumps({
                "search_terms": ["Even", "programming", "Python"],
                "entities": ["Even", "Python"],
                "query_intent": "fact_checking",
                "expanded_terms": ["coding"]
            })),
            # analyze_retrieved_memories
            MagicMock(content=json.dumps({
                "relevant_memories": [
                    {"memory_id": memory_id, "relevance_score": 0.9, "summary": "Answers the query"}
                ],
                "total_count": 1,
                "relevant_count": 1
            })),
            # synthesize_answer
            MagicMock(content="Even loves programming in Python.")
        ]
        
        with patch.object(self.orchestrator, '_classify_intent', return_value='query'):
            result_query = self.orchestrator.process_input(query)
            self.assertEqual(result_query, "Even loves programming in Python.")
        
        # 3. Consolidate Step
        self.mock_llm.chat.side_effect = [
            MagicMock(content=json.dumps({
                "patterns": [],
                "merged_memories": [],
                "new_relations": []
            }))
        ]
        
        with patch.object(self.orchestrator, '_classify_intent', return_value='consolidate'):
            result_consolidate = self.orchestrator.process_input("Please consolidate.")
            self.assertIn("已整合相关记忆！", result_consolidate)

if __name__ == '__main__':
    unittest.main()
