"""Abstract storage backend interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class StorageBackend(ABC):
    """Abstract interface for memory persistence."""

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load all data from storage.

        Returns:
            Dictionary with keys: memories, categories, relationships,
            and optionally: positions, tensions, precedents.
        """
        ...

    @abstractmethod
    def save(self, data: Dict[str, Any]) -> None:
        """Save all data to storage."""
        ...

    @abstractmethod
    def load_index(self, dim: int) -> Optional[Tuple[Any, Dict, Dict]]:
        """Load persisted vector index.

        Args:
            dim: Expected embedding dimension.

        Returns:
            Tuple of (hnsw_index, memory_to_index, index_to_memory) or None.
        """
        ...

    @abstractmethod
    def save_index(self, index: Any, memory_to_index: Dict, index_to_memory: Dict) -> None:
        """Persist the vector index and ID mappings."""
        ...
