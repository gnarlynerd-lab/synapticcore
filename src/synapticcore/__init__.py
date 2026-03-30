"""
SynapticCore — Cognitive memory architecture for LLM agents.

Main entry point: create a SynapticCore instance to access all subsystems.
"""

from .memory.base import MemorySystem
from .memory.enhanced import EnhancedMemory
from .memory.feedback import MemoryFeedbackLoop
from .storage.json_store import JsonFileStore
from .llm.provider import LLMProvider, create_provider


class SynapticCore:
    """
    Facade providing unified access to all SynapticCore subsystems.

    Usage:
        core = SynapticCore(storage_path="memory_store.json")
        core.memory.add_memory("some content", categories=["topic"])
        results = core.memory.enhanced_hybrid_search("query")
        suggestions = core.enhanced.suggest_categories_for_memory(0)
    """

    def __init__(self, storage_path: str = "memory_store.json", llm_provider=None):
        self.storage = JsonFileStore(storage_path)
        self.memory = MemorySystem(storage=self.storage)
        self.enhanced = EnhancedMemory(self.memory)
        self.feedback = MemoryFeedbackLoop(
            self.memory,
            enhanced_memory=self.enhanced,
            feedback_log_path=storage_path.rsplit('.', 1)[0] + '_feedback.json'
            if '.' in storage_path else storage_path + '_feedback.json'
        )
        self.llm = llm_provider


__all__ = [
    "SynapticCore",
    "MemorySystem",
    "EnhancedMemory",
    "MemoryFeedbackLoop",
    "JsonFileStore",
    "LLMProvider",
    "create_provider",
]
