"""
First-class memory types: Position, Tension, Precedent.

These are the intellectual building blocks that SynapticCore tracks.
Unlike generic memories, each type has structured fields, evolution
history, and typed relationships to other objects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import uuid


@dataclass
class Position:
    """What the user holds or has held.

    Tracks confidence evolution and links to related tensions/precedents.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    confidence: str = "tentative"  # tentative | held | committed
    context: str = ""  # what prompted this position
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[List[float]] = None
    categories: List[str] = field(default_factory=list)
    evolution: List[dict] = field(default_factory=list)
    # [{timestamp, previous_confidence, new_confidence, trigger, related_position_id}]
    related_tensions: List[str] = field(default_factory=list)  # tension IDs
    related_precedents: List[str] = field(default_factory=list)  # precedent IDs
    superseded_by: Optional[str] = None  # ID of position that replaced this one
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "context": self.context,
            "timestamp": self.timestamp,
            "embedding": self.embedding,
            "categories": self.categories,
            "evolution": self.evolution,
            "related_tensions": self.related_tensions,
            "related_precedents": self.related_precedents,
            "superseded_by": self.superseded_by,
            "metadata": self.metadata,
        }

    @property
    def is_active(self) -> bool:
        """A position is active if it hasn't been superseded."""
        return self.superseded_by is None

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def update_confidence(self, new_confidence: str, trigger: str = ""):
        """Record a confidence change."""
        self.evolution.append({
            "timestamp": datetime.now().isoformat(),
            "previous_confidence": self.confidence,
            "new_confidence": new_confidence,
            "trigger": trigger,
        })
        self.confidence = new_confidence


@dataclass
class Tension:
    """A recurring unresolved opposition between two poles.

    Tensions are structural features of the user's thinking, not errors.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    poles: List[str] = field(default_factory=list)  # exactly 2
    description: str = ""
    status: str = "active"  # active | explored | dormant | resolved
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[List[float]] = None
    categories: List[str] = field(default_factory=list)
    engagement_history: List[dict] = field(default_factory=list)
    # [{timestamp, context, outcome}]
    adjacent_tensions: List[str] = field(default_factory=list)  # tension IDs
    related_positions: dict = field(default_factory=dict)
    # {pole_index: [position_ids]} — which positions align with each pole
    resolution: Optional[str] = None  # if resolved, what was decided
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "poles": self.poles,
            "description": self.description,
            "status": self.status,
            "timestamp": self.timestamp,
            "embedding": self.embedding,
            "categories": self.categories,
            "engagement_history": self.engagement_history,
            "adjacent_tensions": self.adjacent_tensions,
            "related_positions": self.related_positions,
            "resolution": self.resolution,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Tension":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def record_engagement(self, context: str, outcome: str = ""):
        """Record that this tension was engaged with."""
        self.engagement_history.append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "outcome": outcome,
        })

    def resolve(self, resolution_statement: str):
        """Mark this tension as resolved with the given outcome."""
        self.status = "resolved"
        self.resolution = resolution_statement
        self.engagement_history.append({
            "timestamp": datetime.now().isoformat(),
            "context": "resolved",
            "outcome": resolution_statement,
        })


@dataclass
class Precedent:
    """An established commitment that may be tested.

    Tracks whether the precedent has held under subsequent pressure.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    held: bool = True
    context: str = ""  # under what conditions it was established
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[List[float]] = None
    categories: List[str] = field(default_factory=list)
    test_history: List[dict] = field(default_factory=list)
    # [{timestamp, challenge, outcome, held_after}]
    dependencies: List[str] = field(default_factory=list)  # position IDs that depend on this
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "held": self.held,
            "context": self.context,
            "timestamp": self.timestamp,
            "embedding": self.embedding,
            "categories": self.categories,
            "test_history": self.test_history,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Precedent":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def record_test(self, challenge: str, outcome: str, held_after: bool):
        """Record that this precedent was tested."""
        self.test_history.append({
            "timestamp": datetime.now().isoformat(),
            "challenge": challenge,
            "outcome": outcome,
            "held_after": held_after,
        })
        self.held = held_after
