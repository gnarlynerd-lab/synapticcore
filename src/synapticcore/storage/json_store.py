"""JSON file storage backend."""

import json
import os
from typing import Any, Dict, Optional, Tuple

from .base import StorageBackend


class JsonFileStore(StorageBackend):
    """File-based JSON storage for memory data and HNSW index."""

    def __init__(self, storage_path: str = "memory_store.json"):
        self.storage_path = storage_path

    @property
    def _index_path(self) -> str:
        base = self.storage_path.rsplit('.', 1)[0] if '.' in self.storage_path else self.storage_path
        return base + '.hnsw'

    @property
    def _mapping_path(self) -> str:
        base = self.storage_path.rsplit('.', 1)[0] if '.' in self.storage_path else self.storage_path
        return base + '_index_mapping.json'

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.storage_path):
            return {"memories": [], "categories": {}, "relationships": {}}
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            return {
                "memories": data.get("memories", []),
                "categories": data.get("categories", {}),
                "relationships": data.get("relationships", {}),
                # Future: positions, tensions, precedents
            }
        except Exception as e:
            print(f"Error loading data from {self.storage_path}: {e}")
            return {"memories": [], "categories": {}, "relationships": {}}

    def save(self, data: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")

    def load_index(self, dim: int) -> Optional[Tuple[Any, Dict, Dict]]:
        if not os.path.exists(self._index_path) or not os.path.exists(self._mapping_path):
            return None
        try:
            import hnswlib

            with open(self._mapping_path, 'r') as f:
                mapping_data = json.load(f)

            memory_to_index = {int(k): v for k, v in mapping_data["memory_to_index"].items()}
            index_to_memory = {int(k): v for k, v in mapping_data["index_to_memory"].items()}

            # Load stored max_elements or use a sensible default
            max_elements = max(1000, len(memory_to_index) * 2)
            index = hnswlib.Index(space='cosine', dim=dim)
            index.load_index(self._index_path, max_elements=max_elements)
            index.set_ef(50)

            return index, memory_to_index, index_to_memory
        except Exception as e:
            print(f"Error loading index, will rebuild: {e}")
            return None

    def save_index(self, index: Any, memory_to_index: Dict, index_to_memory: Dict) -> None:
        if index is None or index.get_current_count() == 0:
            return
        try:
            index.save_index(self._index_path)
            mapping_data = {
                "memory_to_index": {str(k): v for k, v in memory_to_index.items()},
                "index_to_memory": {str(k): v for k, v in index_to_memory.items()}
            }
            with open(self._mapping_path, 'w') as f:
                json.dump(mapping_data, f)
        except Exception as e:
            print(f"Error saving index: {e}")
