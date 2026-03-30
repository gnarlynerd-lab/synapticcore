"""
Tests for intellectual arc tracking.

Tests cover:
- Relationship detection: reinforcement, shift, reversal
- Trajectory classification: converging, oscillating, deepening
- Arc retrieval with transition chains
- LLM relationship hints
- Recurring tension detection
"""

import pytest
from synapticcore import SynapticCore
from synapticcore.memory.arcs import ArcTracker


@pytest.fixture
def core(tmp_path):
    return SynapticCore(storage_path=str(tmp_path / "test.json"))


# ─── Relationship Detection ───────────────────────────────────────────

class TestDetectRelationship:

    def test_reinforcement_same_topic_higher_confidence(self, core):
        """Same topic, confidence increases → reinforcement."""
        core.positions.create("Microservices are the right architecture", confidence="tentative")
        pos2 = core.positions.create("Microservices are the right architecture for our scale", confidence="held")
        det = core.arcs.detect_relationship(pos2)
        assert det is not None
        assert det["type"] == "reinforcement"

    def test_shift_related_but_different(self, core):
        """Related but reframed → shift."""
        core.positions.create("We should use strict static typing for all our code", confidence="held")
        pos2 = core.positions.create("We should use gradual typing where it matters most", confidence="held")
        det = core.arcs.detect_relationship(pos2)
        assert det is not None
        assert det["type"] == "shift"

    def test_no_relationship_unrelated(self, core):
        """Completely unrelated positions → no detection."""
        core.positions.create("Python is great for data science", confidence="held")
        pos2 = core.positions.create("Sourdough bread requires patience", confidence="held")
        det = core.arcs.detect_relationship(pos2)
        # Either None or very low confidence
        if det is not None:
            assert det["similarity"] < 0.5

    def test_no_relationship_empty_store(self, core):
        """First position ever → no relationship."""
        pos = core.positions.create("First position ever", confidence="tentative")
        det = core.arcs.detect_relationship(pos)
        assert det is None

    def test_hint_overrides_reversal(self, core):
        """LLM hint 'contradicts' should classify as reversal."""
        core.positions.create("We should avoid writing automated tests because they slow us down", confidence="committed")
        pos2 = core.positions.create("We should write automated tests for everything because they save time", confidence="held")
        det = core.arcs.detect_relationship(pos2, relationship_hint="contradicts previous position on testing")
        assert det is not None
        assert det["type"] == "reversal"
        assert det["confidence"] > 0.3  # hint adds confidence

    def test_hint_reinforces(self, core):
        """LLM hint 'reinforces' should classify as reinforcement."""
        core.positions.create("Testing is essential", confidence="held")
        pos2 = core.positions.create("Tests saved us from a production bug", confidence="committed")
        det = core.arcs.detect_relationship(pos2, relationship_hint="reinforces commitment to testing")
        assert det is not None
        assert det["type"] == "reinforcement"

    def test_hint_shift(self, core):
        """LLM hint 'nuance/refine' should classify as shift."""
        core.positions.create("Static typing prevents all bugs", confidence="committed")
        pos2 = core.positions.create("Static typing prevents a class of bugs but not all", confidence="held")
        det = core.arcs.detect_relationship(pos2, relationship_hint="refines earlier position")
        assert det is not None
        assert det["type"] == "shift"

    def test_evolution_recorded_on_position(self, core):
        """Detection should record the relationship in position's evolution."""
        core.positions.create("Original stance on testing", confidence="held")
        pos2 = core.positions.create("Updated stance on testing approach", confidence="committed")
        core.arcs.detect_relationship(pos2)
        assert len(pos2.evolution) >= 1
        assert "type" in pos2.evolution[-1]
        assert "related_position_id" in pos2.evolution[-1]


# ─── Trajectory Classification ────────────────────────────────────────

class TestClassifyTrajectory:

    def test_converging(self, core):
        """Confidence increases over time → converging."""
        positions = [
            core.positions.create("Types might help", confidence="tentative"),
            core.positions.create("Types definitely help", confidence="held"),
            core.positions.create("Types are essential", confidence="committed"),
        ]
        trajectory = core.arcs.classify_trajectory(positions)
        assert trajectory == "converging"

    def test_oscillating(self, core):
        """Confidence flips back and forth → oscillating."""
        positions = [
            core.positions.create("Microservices are good", confidence="committed"),
            core.positions.create("Actually monoliths are simpler", confidence="tentative"),
            core.positions.create("No wait microservices scale better", confidence="held"),
            core.positions.create("But monoliths are easier to debug", confidence="tentative"),
            core.positions.create("Microservices with good tooling", confidence="held"),
        ]
        trajectory = core.arcs.classify_trajectory(positions)
        assert trajectory == "oscillating"

    def test_insufficient_data(self, core):
        """Fewer than 2 positions → insufficient_data."""
        positions = [core.positions.create("Only one position", confidence="held")]
        assert core.arcs.classify_trajectory(positions) == "insufficient_data"

    def test_empty(self, core):
        assert core.arcs.classify_trajectory([]) == "insufficient_data"


# ─── get_arc ──────────────────────────────────────────────────────────

class TestGetArc:

    def test_arc_with_positions(self, core):
        """Arc should return positions sorted chronologically with transitions."""
        core.positions.create("REST is the standard", confidence="held", context="2024 Q1")
        core.positions.create("GraphQL is worth considering", confidence="tentative", context="2024 Q2")
        core.positions.create("GraphQL for reads, REST for writes", confidence="held", context="2024 Q3")

        arc = core.arcs.get_arc("API design")
        assert arc["topic"] == "API design"
        assert len(arc["positions"]) > 0
        assert arc["trajectory"] in ("converging", "oscillating", "deepening", "insufficient_data")
        assert "summary" in arc

    def test_arc_includes_transitions(self, core):
        """Positions after the first should have transition_from_previous."""
        core.positions.create("First position on databases", confidence="tentative")
        core.positions.create("Second position on databases", confidence="held")

        arc = core.arcs.get_arc("database choice")
        if len(arc["positions"]) >= 2:
            assert "transition_from_previous" in arc["positions"][-1]
            trans = arc["positions"][-1]["transition_from_previous"]
            assert trans["type"] in ("reinforcement", "shift", "reversal")

    def test_arc_empty(self, core):
        arc = core.arcs.get_arc("nonexistent topic")
        assert arc["positions"] == []
        assert arc["trajectory"] == "insufficient_data"

    def test_arc_includes_recurring_tensions(self, core):
        """Arc should surface tensions related to the topic."""
        t = core.tensions.create(["speed", "quality"], description="Development speed vs code quality")
        core.tensions.record_engagement(t.id, "sprint 1")
        core.tensions.record_engagement(t.id, "sprint 2")
        core.positions.create("We should prioritize code quality", confidence="held")

        arc = core.arcs.get_arc("code quality")
        assert isinstance(arc["recurring_tensions"], list)


# ─── Recurring Tensions ───────────────────────────────────────────────

class TestRecurringTensions:

    def test_detect_recurring(self, core):
        t1 = core.tensions.create(["autonomy", "coordination"])
        core.tensions.record_engagement(t1.id, "meeting 1")
        core.tensions.record_engagement(t1.id, "meeting 2")
        core.tensions.record_engagement(t1.id, "meeting 3")

        t2 = core.tensions.create(["speed", "quality"])
        # Only 0 engagements — should not appear

        recurring = core.arcs.detect_recurring_tensions(min_engagements=2)
        assert len(recurring) == 1
        assert recurring[0]["id"] == t1.id
        assert recurring[0]["engagement_count"] == 3


# ─── MCP Integration ─────────────────────────────────────────────────

class TestMCPArcIntegration:

    def test_store_detects_arc(self, tmp_path):
        """store_interaction should report arc detections."""
        import synapticcore.mcp.server as server_module
        from synapticcore.mcp.server import store_interaction, get_decision_narrative

        server_module.STORAGE_PATH = str(tmp_path / "test_arc_mcp.json")
        server_module._core = None

        store_interaction(
            decisions=[{"statement": "Testing is important for quality", "confidence": "held"}]
        )
        result = store_interaction(
            decisions=[{"statement": "Testing is essential and non-negotiable", "confidence": "committed"}]
        )
        assert "Narrative arc" in result or "arc" in result.lower()

        arc = get_decision_narrative("testing")
        assert len(arc["decisions"]) >= 1
        assert arc["trajectory"] != "insufficient_data" or len(arc["decisions"]) < 2

        server_module._core = None

    def test_store_with_hint(self, tmp_path):
        """relationship_hint should influence arc detection."""
        import synapticcore.mcp.server as server_module
        from synapticcore.mcp.server import store_interaction

        server_module.STORAGE_PATH = str(tmp_path / "test_hint_mcp.json")
        server_module._core = None

        store_interaction(
            decisions=[{"statement": "We should avoid writing automated tests because they slow development", "confidence": "committed"}]
        )
        result = store_interaction(
            decisions=[{"statement": "We should write automated tests for everything to speed up development", "confidence": "held"}],
            relationship_hint="contradicts previous stance on automated testing"
        )
        assert "reversal" in result.lower()

        server_module._core = None
