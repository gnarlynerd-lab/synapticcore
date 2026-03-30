"""
SynapticCore — Cognitive memory architecture for LLM agents.

Main entry point: create a SynapticCore instance to access all subsystems.
"""

from .memory.base import MemorySystem
from .memory.enhanced import EnhancedMemory
from .memory.feedback import MemoryFeedbackLoop
from .memory.types import Position, Tension, Precedent
from .memory.type_managers import PositionManager, TensionManager, PrecedentManager
from .storage.json_store import JsonFileStore
from .llm.provider import LLMProvider, create_provider


class SynapticCore:
    """
    Facade providing unified access to all SynapticCore subsystems.

    Usage:
        core = SynapticCore(storage_path="memory_store.json")
        core.positions.create("Types prevent bugs", confidence="held")
        core.tensions.create(["type safety", "velocity"])
        results = core.memory.enhanced_hybrid_search("query")
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

        # First-class memory types
        self.positions = PositionManager(self.memory)
        self.tensions = TensionManager(self.memory)
        self.precedents = PrecedentManager(self.memory)

        # Load typed data from storage
        self._load_types()

    def _load_types(self):
        """Load positions, tensions, precedents from storage."""
        data = self.storage.load()
        self.positions.load_from_dicts(data.get("positions", []))
        self.tensions.load_from_dicts(data.get("tensions", []))
        self.precedents.load_from_dicts(data.get("precedents", []))

    def save(self):
        """Save all data including typed memories."""
        # Base memory system saves its own data
        self.memory._save_data()
        # Overlay typed data onto the store
        data = self.storage.load()
        data["positions"] = self.positions.to_dicts()
        data["tensions"] = self.tensions.to_dicts()
        data["precedents"] = self.precedents.to_dicts()
        self.storage.save(data)


__all__ = [
    "SynapticCore",
    "MemorySystem",
    "EnhancedMemory",
    "MemoryFeedbackLoop",
    "JsonFileStore",
    "LLMProvider",
    "create_provider",
]
