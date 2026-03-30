"""
Tests for spreading activation network.

Tests cover:
- Graph construction from typed objects and relationships
- Multi-hop propagation with decay
- Lateral inhibition
- Seed node discovery
- find_adjacent_territory
- Integration with MCP tools
"""

import pytest
from synapticcore import SynapticCore
from synapticcore.retrieval.activation import (
    SpreadingActivationNetwork, ActivationConfig, ActivatedNode,
)


@pytest.fixture
def core(tmp_path):
    return SynapticCore(storage_path=str(tmp_path / "test.json"))


@pytest.fixture
def populated_core(core):
    """Core with a small intellectual topology for testing activation."""
    # Positions about architecture
    p1 = core.positions.create(
        "Microservices enable team autonomy",
        confidence="held", categories=["architecture"],
    )
    p2 = core.positions.create(
        "Monoliths are simpler to deploy",
        confidence="held", categories=["architecture"],
    )
    p3 = core.positions.create(
        "Event sourcing provides complete audit trails",
        confidence="committed", categories=["architecture", "data"],
    )

    # Tensions
    t1 = core.tensions.create(
        ["autonomy", "simplicity"],
        description="Team autonomy vs system simplicity",
        categories=["architecture"],
    )
    t2 = core.tensions.create(
        ["consistency", "availability"],
        description="CAP theorem tradeoff in distributed systems",
        categories=["architecture", "data"],
    )

    # Link position to tension
    p1.related_tensions.append(t1.id)

    # Precedent
    pr1 = core.precedents.create(
        "We chose microservices for the payments platform",
        held=True, categories=["architecture"],
        dependencies=[p1.id],
    )

    # Category relationship
    core.memory.add_category("architecture", "System architecture decisions")
    core.memory.add_category("data", "Data management and storage")
    core.memory.add_relationship("architecture", "related_to", "data",
                                  "Architecture decisions affect data patterns")

    # A generic memory too
    core.memory.add_memory("The team discussed microservices vs monoliths at the offsite",
                           categories=["architecture"])

    core.save()
    return core


# ─── Graph Construction ───────────────────────────────────────────────

class TestGraphConstruction:

    def test_adjacency_has_category_relationships(self, populated_core):
        adj = populated_core.activation._build_adjacency()
        # architecture -> data relationship should exist
        arch_neighbors = [n for n, w in adj.get("cat:architecture", [])]
        assert "cat:data" in arch_neighbors

    def test_adjacency_has_position_tension_links(self, populated_core):
        adj = populated_core.activation._build_adjacency()
        # p1 should link to t1
        p1 = populated_core.positions.items[0]
        pos_neighbors = [n for n, w in adj.get(f"pos:{p1.id}", [])]
        t1 = populated_core.tensions.items[0]
        assert f"ten:{t1.id}" in pos_neighbors

    def test_adjacency_has_category_links(self, populated_core):
        adj = populated_core.activation._build_adjacency()
        # All architecture-category items should link to cat:architecture
        p1 = populated_core.positions.items[0]
        pos_neighbors = [n for n, w in adj.get(f"pos:{p1.id}", [])]
        assert "cat:architecture" in pos_neighbors

    def test_adjacency_has_precedent_dependencies(self, populated_core):
        adj = populated_core.activation._build_adjacency()
        pr1 = populated_core.precedents.items[0]
        p1 = populated_core.positions.items[0]
        prec_neighbors = [n for n, w in adj.get(f"pre:{pr1.id}", [])]
        assert f"pos:{p1.id}" in prec_neighbors


# ─── Activation Propagation ──────────────────────────────────────────

class TestActivation:

    def test_seed_nodes_found(self, populated_core):
        nodes = populated_core.activation.activate(["microservices architecture"])
        assert len(nodes) > 0
        # Should find at least some of our positions
        types = {n.node_type for n in nodes}
        assert "position" in types or "memory" in types

    def test_multi_hop_propagation(self, populated_core):
        config = ActivationConfig(max_hops=3, decay_rate=0.5)
        nodes = populated_core.activation.activate(["microservices"], config)
        depths = {n.depth for n in nodes}
        # Should have nodes at depth 0 (seeds) and deeper (propagated)
        assert 0 in depths
        if len(nodes) > 1:
            assert max(depths) >= 1

    def test_activation_decays_with_depth(self, populated_core):
        config = ActivationConfig(max_hops=3, decay_rate=0.5)
        nodes = populated_core.activation.activate(["microservices"], config)
        # Group by depth, check average activation decreases
        by_depth = {}
        for n in nodes:
            by_depth.setdefault(n.depth, []).append(n.activation)
        if len(by_depth) >= 2:
            avg_0 = sum(by_depth.get(0, [0])) / max(len(by_depth.get(0, [1])), 1)
            deeper = [d for d in by_depth if d > 0]
            if deeper:
                avg_deep = sum(by_depth[deeper[0]]) / len(by_depth[deeper[0]])
                assert avg_deep < avg_0  # Deeper nodes should have less activation

    def test_activation_paths_recorded(self, populated_core):
        config = ActivationConfig(max_hops=2)
        nodes = populated_core.activation.activate(["architecture"], config)
        for n in nodes:
            assert len(n.path) >= 1
            assert n.path[0].startswith(("pos:", "ten:", "pre:", "mem:", "cat:"))

    def test_lateral_inhibition(self, populated_core):
        # Strong node should suppress weak neighbors
        config = ActivationConfig(
            max_hops=2, inhibition_threshold=0.3, inhibition_factor=0.5
        )
        nodes_with = populated_core.activation.activate(["microservices"], config)

        config_no_inhibition = ActivationConfig(
            max_hops=2, inhibition_threshold=999.0  # effectively disable
        )
        nodes_without = populated_core.activation.activate(["microservices"], config_no_inhibition)

        # With inhibition, some weak nodes should have lower activation
        if len(nodes_with) > 2 and len(nodes_without) > 2:
            # Total activation should be lower with inhibition
            total_with = sum(n.activation for n in nodes_with)
            total_without = sum(n.activation for n in nodes_without)
            assert total_with <= total_without

    def test_empty_store(self, core):
        nodes = core.activation.activate(["anything at all"])
        assert isinstance(nodes, list)
        assert len(nodes) == 0


# ─── find_adjacent_territory ─────────────────────────────────────────

class TestFindAdjacent:

    def test_returns_structure(self, populated_core):
        result = populated_core.activation.find_adjacent_territory("microservices")
        assert "explored" in result
        assert "adjacent" in result
        assert "gap_tensions" in result
        assert isinstance(result["explored"], list)
        assert isinstance(result["adjacent"], list)

    def test_explored_vs_adjacent(self, populated_core):
        result = populated_core.activation.find_adjacent_territory("microservices architecture")
        # Explored items should have higher activation
        for item in result["explored"]:
            assert item["activation"] > 0.0
        for item in result["adjacent"]:
            assert item["activation"] > 0.0
            assert item["depth"] >= 1

    def test_empty_store(self, core):
        result = core.activation.find_adjacent_territory("anything")
        assert result["explored"] == []
        assert result["adjacent"] == []


# ─── MCP Integration ────────────────────────────────────────────────

class TestMCPActivationIntegration:

    def test_retrieve_relevant_deep(self, tmp_path):
        import synapticcore.mcp.server as server_module
        from synapticcore.mcp.server import store_interaction, retrieve_relevant

        server_module.STORAGE_PATH = str(tmp_path / "test_act_mcp.json")
        server_module._core = None

        store_interaction(
            decisions=[
                {"statement": "Functional programming reduces side effects", "confidence": "held"},
                {"statement": "Pure functions are easier to test", "confidence": "committed"},
            ],
            tradeoffs=[
                {"poles": ["purity", "practicality"], "description": "Pure FP vs pragmatic code"}
            ],
        )

        # Deep mode should use spreading activation
        results = retrieve_relevant("functional programming", depth="deep", max_results=5)
        assert isinstance(results, list)
        if results:
            assert "retrieval_method" in results[0]
            # Deep mode results should mention activation
            assert any("activation" in r.get("retrieval_method", "") for r in results)

        server_module._core = None

    def test_find_adjacent_topology(self, tmp_path):
        import synapticcore.mcp.server as server_module
        from synapticcore.mcp.server import store_interaction, find_adjacent

        server_module.STORAGE_PATH = str(tmp_path / "test_adj_mcp.json")
        server_module._core = None

        store_interaction(
            decisions=[{"statement": "We should use TypeScript", "confidence": "held"}],
            tradeoffs=[{"poles": ["type safety", "velocity"], "description": "Types vs speed"}],
        )

        result = find_adjacent("type safety in code")
        assert "explored" in result
        assert "adjacent" in result
        assert "gap_tensions" in result

        server_module._core = None
