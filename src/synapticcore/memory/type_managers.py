"""
Managers for first-class memory types.

Each manager handles CRUD, embedding, and type-specific operations for
positions, tensions, or precedents. They share the MemorySystem's HNSW
index using type-prefixed IDs.
"""

import logging
from typing import List, Dict, Optional, Type, TypeVar
import numpy as np

from .base import MemorySystem
from .types import Position, Tension, Precedent

logger = logging.getLogger(__name__)

T = TypeVar("T", Position, Tension, Precedent)


class _TypeManager:
    """Base manager for a typed memory collection.

    Handles embedding, index integration, and search. Subclasses
    add type-specific operations.
    """

    type_prefix: str = ""  # e.g. "pos", "ten", "pre"
    type_class: Type = None
    text_field: str = "statement"  # field to embed

    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system
        self.items: List = []  # List of typed objects
        # Index mappings: item list index -> HNSW index ID
        self._item_to_index: Dict[int, int] = {}
        self._index_to_item: Dict[int, int] = {}

    def _get_text(self, item) -> str:
        """Get the text to embed for an item."""
        return getattr(item, self.text_field, "")

    def _embed(self, text: str) -> Optional[List[float]]:
        """Create embedding using the shared model."""
        if not self.memory_system.embedding_model:
            return None
        try:
            return self.memory_system.embedding_model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None

    def _add_to_shared_index(self, item_idx: int, embedding: List[float]):
        """Add an embedding to the shared HNSW index with a type-prefixed ID."""
        if not self.memory_system.vector_index:
            return
        emb = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Use the current total count in the HNSW index as the next ID
        # This avoids collisions across memories and all type managers
        index_id = self.memory_system.vector_index.get_current_count()
        # Ensure we don't exceed max_elements — resize if needed
        current_max = self.memory_system.vector_index.get_max_elements()
        current_count = self.memory_system.vector_index.get_current_count()
        if current_count >= current_max - 1:
            self.memory_system.vector_index.resize_index(current_max * 2)

        self.memory_system.vector_index.add_items(emb, index_id)
        self._item_to_index[item_idx] = index_id
        self._index_to_item[index_id] = item_idx

    def add(self, item) -> int:
        """Add a typed item. Returns its index."""
        # Embed
        text = self._get_text(item)
        if text and not item.embedding:
            item.embedding = self._embed(text)

        item_idx = len(self.items)
        self.items.append(item)

        if item.embedding and self.memory_system.vector_index:
            self._add_to_shared_index(item_idx, item.embedding)

        return item_idx

    def get(self, idx: int):
        """Get item by index."""
        if 0 <= idx < len(self.items):
            return self.items[idx]
        return None

    def get_by_id(self, item_id: str):
        """Get item by UUID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Semantic search within this type's items."""
        if not self.memory_system.embedding_model or not self.memory_system.vector_index:
            return []
        if not self.items:
            return []

        query_emb = self.memory_system.embedding_model.encode(query)
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # Search the full index, then filter to our type's IDs
        our_index_ids = set(self._item_to_index.values())
        if not our_index_ids:
            return []

        count = self.memory_system.vector_index.get_current_count()
        if count == 0:
            return []

        labels, distances = self.memory_system.vector_index.knn_query(
            query_emb, k=min(top_k * 5, count)
        )

        results = []
        for idx, dist in zip(labels[0], distances[0]):
            idx = int(idx)
            if idx not in our_index_ids:
                continue
            item_idx = self._index_to_item[idx]
            item = self.items[item_idx]
            results.append({
                "item": item,
                "index": item_idx,
                "similarity": float(1.0 - dist),
            })
            if len(results) >= top_k:
                break

        return results

    def to_dicts(self) -> List[dict]:
        """Serialize all items for storage."""
        return [item.to_dict() for item in self.items]

    def load_from_dicts(self, data: List[dict]):
        """Load items from stored dicts."""
        for d in data:
            item = self.type_class.from_dict(d)
            item_idx = len(self.items)
            self.items.append(item)
            if item.embedding and self.memory_system.vector_index:
                self._add_to_shared_index(item_idx, item.embedding)


class PositionManager(_TypeManager):
    type_prefix = "pos"
    type_class = Position
    text_field = "statement"

    def create(self, statement: str, confidence: str = "tentative",
               context: str = "", categories: List[str] = None,
               related_tensions: List[str] = None,
               related_precedents: List[str] = None,
               metadata: dict = None) -> Position:
        """Create and store a new position."""
        pos = Position(
            statement=statement,
            confidence=confidence,
            context=context,
            categories=categories or [],
            related_tensions=related_tensions or [],
            related_precedents=related_precedents or [],
            metadata=metadata or {},
        )
        self.add(pos)
        return pos

    def update_confidence(self, position_id: str, new_confidence: str,
                          trigger: str = "") -> Optional[Position]:
        """Update a position's confidence level."""
        pos = self.get_by_id(position_id)
        if pos:
            pos.update_confidence(new_confidence, trigger)
        return pos

    def find_related(self, position_id: str) -> dict:
        """Find tensions and precedents linked to a position."""
        pos = self.get_by_id(position_id)
        if not pos:
            return {"tensions": [], "precedents": []}
        return {
            "tensions": pos.related_tensions,
            "precedents": pos.related_precedents,
        }


class TensionManager(_TypeManager):
    type_prefix = "ten"
    type_class = Tension
    text_field = "description"

    def create(self, poles: List[str], description: str = "",
               status: str = "active", categories: List[str] = None,
               metadata: dict = None) -> Tension:
        """Create and store a new tension."""
        if not description:
            description = " vs ".join(poles)
        ten = Tension(
            poles=poles,
            description=description,
            status=status,
            categories=categories or [],
            metadata=metadata or {},
        )
        self.add(ten)
        return ten

    def record_engagement(self, tension_id: str, context: str,
                          outcome: str = "") -> Optional[Tension]:
        """Record that a tension was engaged with."""
        ten = self.get_by_id(tension_id)
        if ten:
            ten.record_engagement(context, outcome)
        return ten

    def get_recurring(self, min_engagements: int = 2) -> List[Tension]:
        """Find tensions that keep surfacing."""
        return [t for t in self.items
                if len(t.engagement_history) >= min_engagements and t.status == "active"]


class PrecedentManager(_TypeManager):
    type_prefix = "pre"
    type_class = Precedent
    text_field = "statement"

    def create(self, statement: str, held: bool = True,
               context: str = "", categories: List[str] = None,
               dependencies: List[str] = None,
               metadata: dict = None) -> Precedent:
        """Create and store a new precedent."""
        prec = Precedent(
            statement=statement,
            held=held,
            context=context,
            categories=categories or [],
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        self.add(prec)
        return prec

    def record_test(self, precedent_id: str, challenge: str,
                    outcome: str, held_after: bool) -> Optional[Precedent]:
        """Record that a precedent was tested."""
        prec = self.get_by_id(precedent_id)
        if prec:
            prec.record_test(challenge, outcome, held_after)
        return prec

    def get_broken(self) -> List[Precedent]:
        """Find precedents that have broken."""
        return [p for p in self.items if not p.held]
