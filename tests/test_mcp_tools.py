"""
Tests for MCP tool handlers.

Tests the tool functions directly — they're regular functions backed by SynapticCore.
"""

import os
import pytest

# Override storage path before importing server
os.environ["SYNAPTICCORE_STORAGE"] = ""  # Will be overridden per-test


from synapticcore.mcp.server import (
    store_interaction,
    retrieve_relevant,
    get_decision_narrative,
    find_adjacent,
    check_constraints,
    assess_depth,
    get_core,
    mcp,
)
import synapticcore.mcp.server as server_module


@pytest.fixture(autouse=True)
def fresh_core(tmp_path):
    """Reset global core for each test with a temp store."""
    storage_path = str(tmp_path / "test_mcp.json")
    server_module.STORAGE_PATH = storage_path
    server_module._core = None
    yield
    server_module._core = None


# ─── Tool Registration ─────────────────────────────────────────────────

class TestToolRegistration:

    def test_all_tools_registered(self):
        tools = mcp._tool_manager._tools
        expected = {"store_interaction", "retrieve_relevant", "get_decision_narrative",
                    "find_adjacent", "check_constraints", "assess_depth"}
        assert expected == set(tools.keys())


# ─── store_interaction ──────────────────────────────────────────────────

class TestStoreInteraction:

    def test_store_decisions(self):
        result = store_interaction(
            decisions=[
                {"statement": "Decentralization has practical limits", "confidence": "held", "context": "discussing system architecture"}
            ]
        )
        assert "decision:" in result
        core = get_core()
        assert len(core.positions.items) == 1
        assert core.positions.items[0].confidence == "held"

    def test_store_tradeoffs(self):
        result = store_interaction(
            tradeoffs=[
                {"poles": ["autonomy", "coordination"], "description": "tradeoff between independence and teamwork", "status": "active"}
            ]
        )
        assert "tradeoff:" in result
        core = get_core()
        assert len(core.tensions.items) == 1
        assert core.tensions.items[0].poles == ["autonomy", "coordination"]

    def test_store_constraints(self):
        result = store_interaction(
            constraints=[
                {"statement": "We always test before deploying", "held": True, "context": "post-incident review"}
            ]
        )
        assert "constraint:" in result
        core = get_core()
        assert len(core.precedents.items) == 1
        assert core.precedents.items[0].held is True

    def test_store_mixed(self):
        result = store_interaction(
            decisions=[{"statement": "Types are worth the overhead", "confidence": "committed", "context": "code review"}],
            tradeoffs=[{"poles": ["type safety", "velocity"], "description": "tradeoff", "status": "active"}],
            constraints=[{"statement": "Use TypeScript for new projects", "held": True, "context": "team decision"}],
            session_summary="Discussed type safety tradeoffs"
        )
        assert "decision:" in result
        assert "tradeoff:" in result
        assert "constraint:" in result
        core = get_core()
        assert len(core.positions.items) == 1
        assert len(core.tensions.items) == 1
        assert len(core.precedents.items) == 1
        assert len(core.memory.memories) == 1  # just the session summary

    def test_store_empty(self):
        result = store_interaction()
        assert "Stored 0 items" in result


# ─── retrieve_relevant ──────────────────────────────────────────────────

class TestRetrieveRelevant:

    def test_retrieve_after_store(self):
        store_interaction(
            decisions=[
                {"statement": "Python is excellent for prototyping", "confidence": "held", "context": "language comparison"},
                {"statement": "Rust is better for production systems", "confidence": "tentative", "context": "language comparison"}
            ]
        )
        results = retrieve_relevant("programming language choice", depth="structural", max_results=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]
        assert "score" in results[0]

    def test_retrieve_empty(self):
        results = retrieve_relevant("anything at all")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_retrieve_depth_levels(self):
        store_interaction(
            decisions=[{"statement": "Test driven development works", "confidence": "committed", "context": "methodology"}]
        )
        surface = retrieve_relevant("testing methodology", depth="surface")
        deep = retrieve_relevant("testing methodology", depth="deep")
        assert isinstance(surface, list)
        assert isinstance(deep, list)

    def test_retrieve_includes_type_metadata(self):
        store_interaction(
            decisions=[{"statement": "Microservices add complexity", "confidence": "held", "context": "arch review"}]
        )
        results = retrieve_relevant("microservices architecture")
        if results:
            assert results[0]["type"] == "decision"
            assert "confidence" in results[0]


# ─── get_intellectual_arc ───────────────────────────────────────────────

class TestGetDecisionNarrative:

    def test_narrative_returns_structure(self):
        store_interaction(
            decisions=[
                {"statement": "Monoliths are simpler", "confidence": "held", "context": "early career"},
                {"statement": "Microservices scale better", "confidence": "tentative", "context": "growth phase"},
            ]
        )
        arc = get_decision_narrative("architecture choices")
        assert "topic" in arc
        assert "decisions" in arc
        assert "trajectory" in arc
        assert "recurring_tradeoffs" in arc
        assert isinstance(arc["decisions"], list)

    def test_narrative_empty_store(self):
        arc = get_decision_narrative("anything")
        assert arc["decisions"] == []
        assert arc["trajectory"] == "insufficient_data"


# ─── find_adjacent ──────────────────────────────────────────────────────

class TestFindAdjacent:

    def test_find_adjacent_returns_structure(self):
        store_interaction(
            decisions=[{"statement": "Functional programming reduces bugs", "confidence": "held", "context": "code quality"}]
        )
        result = find_adjacent("functional programming")
        assert "explored" in result
        assert "adjacent" in result
        assert "gap_tensions" in result

    def test_find_adjacent_empty(self):
        result = find_adjacent("anything")
        assert isinstance(result["explored"], list)
        assert isinstance(result["adjacent"], list)


# ─── check_precedent ───────────────────────────────────────────────────

class TestCheckConstraints:

    def test_finds_constraints(self):
        store_interaction(
            constraints=[
                {"statement": "Never deploy on Friday", "held": True, "context": "incident retro"}
            ]
        )
        results = check_constraints("deployment timing")
        assert isinstance(results, list)
        if results:
            assert "statement" in results[0]
            assert "held" in results[0]

    def test_finds_committed_decisions_as_constraints(self):
        store_interaction(
            decisions=[{"statement": "Always write tests first", "confidence": "committed", "context": "team norm"}]
        )
        results = check_constraints("testing practices")
        assert isinstance(results, list)


# ─── assess_depth ───────────────────────────────────────────────────────

class TestAssessDepth:

    def test_assess_with_structure(self):
        store_interaction(
            decisions=[{"statement": "Types prevent bugs", "confidence": "held", "context": "quality discussion"}],
            tradeoffs=[{"poles": ["type safety", "development speed"], "description": "tradeoff", "status": "active"}]
        )
        result = assess_depth(
            recent_exchange="We were discussing whether to add types to this codebase",
            current_topic="type safety tradeoffs"
        )
        assert "has_deeper_structure" in result
        assert "suggestion" in result
        assert "narrative_material" in result

    def test_assess_empty(self):
        result = assess_depth(recent_exchange="hello", current_topic="greeting")
        assert result["has_deeper_structure"] is False


# ─── Round-trip Integration ────────────────────────────────────────────

class TestRoundTrip:

    def test_store_then_retrieve_then_narrative(self):
        """Full round-trip: store interaction, retrieve it, trace the narrative."""
        store_interaction(
            decisions=[
                {"statement": "Event sourcing is worth the complexity for audit trails",
                 "confidence": "held", "context": "discussing data architecture"}
            ],
            tradeoffs=[
                {"poles": ["simplicity", "auditability"],
                 "description": "Simpler systems vs complete audit history",
                 "status": "active"}
            ],
            constraints=[
                {"statement": "We chose event sourcing for the payments system",
                 "held": True, "context": "Q1 architecture decision"}
            ],
            session_summary="Discussed event sourcing tradeoffs for new service"
        )

        results = retrieve_relevant("data architecture decisions", depth="structural")
        assert len(results) > 0

        arc = get_decision_narrative("event sourcing")
        assert len(arc["decisions"]) > 0 or len(arc.get("related_memories", [])) > 0

        constraints = check_constraints("architecture decisions")
        assert isinstance(constraints, list)

        depth = assess_depth(
            recent_exchange="Should we use event sourcing for the new notification service?",
            current_topic="event sourcing"
        )
        assert depth["has_deeper_structure"] is True
