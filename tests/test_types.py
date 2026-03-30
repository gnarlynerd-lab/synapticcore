"""
Tests for first-class memory types: Position, Tension, Precedent.

Tests cover data models, managers, storage persistence, and HNSW index sharing.
"""

import os
import pytest
import numpy as np

from synapticcore import SynapticCore
from synapticcore.memory.types import Position, Tension, Precedent
from synapticcore.memory.type_managers import PositionManager, TensionManager, PrecedentManager


@pytest.fixture
def core(tmp_path):
    return SynapticCore(storage_path=str(tmp_path / "test.json"))


# ─── Data Model Tests ──────────────────────────────────────────────────

class TestPosition:

    def test_create(self):
        p = Position(statement="Types prevent bugs", confidence="held", context="code review")
        assert p.statement == "Types prevent bugs"
        assert p.confidence == "held"
        assert p.id  # UUID generated

    def test_update_confidence(self):
        p = Position(statement="test", confidence="tentative")
        p.update_confidence("held", trigger="evidence from production")
        assert p.confidence == "held"
        assert len(p.evolution) == 1
        assert p.evolution[0]["previous_confidence"] == "tentative"
        assert p.evolution[0]["new_confidence"] == "held"

    def test_round_trip(self):
        p = Position(statement="test", confidence="committed", context="team decision")
        d = p.to_dict()
        p2 = Position.from_dict(d)
        assert p2.statement == "test"
        assert p2.confidence == "committed"
        assert p2.id == p.id


class TestTension:

    def test_create(self):
        t = Tension(poles=["autonomy", "coordination"], description="independence vs teamwork")
        assert t.poles == ["autonomy", "coordination"]
        assert t.status == "active"

    def test_record_engagement(self):
        t = Tension(poles=["speed", "quality"])
        t.record_engagement("sprint planning", "chose quality over speed")
        assert len(t.engagement_history) == 1

    def test_round_trip(self):
        t = Tension(poles=["a", "b"], status="explored")
        t2 = Tension.from_dict(t.to_dict())
        assert t2.poles == ["a", "b"]
        assert t2.id == t.id


class TestPrecedent:

    def test_create(self):
        p = Precedent(statement="Never deploy on Friday", held=True, context="incident retro")
        assert p.held is True

    def test_record_test(self):
        p = Precedent(statement="test", held=True)
        p.record_test("pressure to ship", "deployed anyway", held_after=False)
        assert p.held is False
        assert len(p.test_history) == 1
        assert p.test_history[0]["held_after"] is False

    def test_round_trip(self):
        p = Precedent(statement="test", held=True)
        p2 = Precedent.from_dict(p.to_dict())
        assert p2.statement == "test"
        assert p2.id == p.id


# ─── Manager Tests ─────────────────────────────────────────────────────

class TestPositionManager:

    def test_create_and_get(self, core):
        pos = core.positions.create("Types prevent bugs", confidence="held", context="review")
        assert pos.statement == "Types prevent bugs"
        assert pos.embedding is not None
        retrieved = core.positions.get_by_id(pos.id)
        assert retrieved is pos

    def test_search(self, core):
        core.positions.create("Python is great for prototyping", confidence="held")
        core.positions.create("Rust is better for production", confidence="tentative")
        core.positions.create("Sourdough takes patience", confidence="held")

        results = core.positions.search("programming language choice", top_k=2)
        assert len(results) > 0
        # Programming positions should rank higher than cooking
        assert "Python" in results[0]["item"].statement or "Rust" in results[0]["item"].statement

    def test_update_confidence(self, core):
        pos = core.positions.create("Microservices scale better", confidence="tentative")
        core.positions.update_confidence(pos.id, "committed", trigger="production experience")
        assert pos.confidence == "committed"
        assert len(pos.evolution) == 1


class TestTensionManager:

    def test_create_and_search(self, core):
        core.tensions.create(["type safety", "development speed"], description="Types slow you down vs prevent bugs")
        core.tensions.create(["autonomy", "coordination"], description="Independence vs teamwork")

        results = core.tensions.search("type checking tradeoffs", top_k=2)
        assert len(results) > 0

    def test_recurring(self, core):
        t = core.tensions.create(["speed", "quality"])
        core.tensions.record_engagement(t.id, "sprint 1")
        core.tensions.record_engagement(t.id, "sprint 2")
        recurring = core.tensions.get_recurring(min_engagements=2)
        assert len(recurring) == 1
        assert recurring[0].id == t.id


class TestPrecedentManager:

    def test_create_and_search(self, core):
        core.precedents.create("Never deploy on Friday", held=True, context="incident retro")
        results = core.precedents.search("deployment timing", top_k=2)
        assert len(results) > 0

    def test_record_test_and_broken(self, core):
        p = core.precedents.create("Always write tests first", held=True)
        core.precedents.record_test(p.id, "deadline pressure", "skipped tests", held_after=False)
        broken = core.precedents.get_broken()
        assert len(broken) == 1
        assert broken[0].id == p.id


# ─── Storage Persistence ──────────────────────────────────────────────

class TestTypePersistence:

    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "persist.json")
        core1 = SynapticCore(storage_path=path)
        core1.positions.create("Test position", confidence="held")
        core1.tensions.create(["a", "b"], description="test tension")
        core1.precedents.create("Test precedent", held=True)
        core1.save()

        core2 = SynapticCore(storage_path=path)
        assert len(core2.positions.items) == 1
        assert core2.positions.items[0].statement == "Test position"
        assert len(core2.tensions.items) == 1
        assert core2.tensions.items[0].poles == ["a", "b"]
        assert len(core2.precedents.items) == 1
        assert core2.precedents.items[0].held is True

    def test_backward_compat(self, tmp_path):
        """Loading old JSON without typed keys should work."""
        import json
        path = str(tmp_path / "old.json")
        # Write old-format store
        with open(path, 'w') as f:
            json.dump({"memories": [], "categories": {}, "relationships": {}}, f)

        core = SynapticCore(storage_path=path)
        assert len(core.positions.items) == 0
        assert len(core.tensions.items) == 0
        assert len(core.precedents.items) == 0

        # Should be able to add new types
        core.positions.create("New position")
        core.save()

        core2 = SynapticCore(storage_path=path)
        assert len(core2.positions.items) == 1


# ─── Shared Index Tests ───────────────────────────────────────────────

class TestSharedIndex:

    def test_types_share_index_with_memories(self, core):
        """Typed objects and generic memories coexist in the same HNSW index."""
        core.memory.add_memory("Python programming language", categories=["tech"])
        core.positions.create("Python is excellent for ML", confidence="held")
        core.tensions.create(["Python", "Rust"], description="Language choice tension")

        # All should be searchable through the shared index
        assert core.memory.vector_index.get_current_count() >= 3

    def test_typed_search_doesnt_return_memories(self, core):
        """Position search shouldn't return generic memories."""
        core.memory.add_memory("Generic memory about cooking")
        core.positions.create("Types prevent bugs", confidence="held")

        results = core.positions.search("cooking", top_k=5)
        # Should only find positions, not generic memories
        for r in results:
            assert isinstance(r["item"], Position)
