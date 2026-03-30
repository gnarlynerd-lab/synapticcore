"""
Spreading activation network for topology-aware retrieval.

Replaces naive category-overlap expansion with proper spreading activation
dynamics. Activation propagates through a graph built from:
- Category relationships
- Position → tension links
- Position → precedent links
- Tension adjacency
- Semantic similarity edges (from HNSW nearest neighbors)

Features:
- Multi-hop propagation with configurable decay
- Lateral inhibition: strongly activated nodes suppress weak neighbors
- Temporal decay: old connections contribute less
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActivationConfig:
    """Configuration for spreading activation dynamics."""
    decay_rate: float = 0.5         # Activation multiplier per hop
    inhibition_threshold: float = 0.6  # Above this, a node suppresses weak neighbors
    inhibition_factor: float = 0.3  # How much strong nodes suppress weak ones
    max_hops: int = 3               # Maximum propagation depth
    temporal_half_life_days: float = 30.0  # Connection age decay
    min_activation: float = 0.05    # Below this, stop propagating
    semantic_neighbor_k: int = 5    # Number of HNSW neighbors per node


@dataclass
class ActivatedNode:
    """A node in the activation network with its current state."""
    node_id: str           # Unique ID (e.g. "pos:uuid", "ten:uuid", "cat:name", "mem:idx")
    node_type: str         # "position", "tension", "precedent", "category", "memory"
    activation: float      # Current activation level [0, 1]
    depth: int             # How many hops from a seed node
    path: List[str]        # IDs of nodes traversed to reach this one
    content: str = ""      # Display text
    metadata: dict = field(default_factory=dict)


class SpreadingActivationNetwork:
    """
    Builds and propagates activation through the intellectual topology.

    The graph has nodes for positions, tensions, precedents, categories,
    and optionally generic memories. Edges come from explicit relationships
    (position→tension, category relationships) and implicit similarity
    (HNSW nearest neighbors).
    """

    def __init__(self, memory_system, positions, tensions, precedents):
        """
        Args:
            memory_system: MemorySystem instance (for categories, relationships, embeddings)
            positions: PositionManager
            tensions: TensionManager
            precedents: PrecedentManager
        """
        self.memory_system = memory_system
        self.positions = positions
        self.tensions = tensions
        self.precedents = precedents

    def _build_adjacency(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Build the adjacency list for the activation graph.

        Returns:
            {node_id: [(neighbor_id, edge_weight), ...]}
        """
        adj: Dict[str, List[Tuple[str, float]]] = {}

        def add_edge(a: str, b: str, weight: float):
            adj.setdefault(a, []).append((b, weight))
            adj.setdefault(b, []).append((a, weight))

        # 1. Category relationships
        for rel_id, rel in self.memory_system.relationships.items():
            if rel.get("status") != "active":
                continue
            src = f"cat:{rel['source']}"
            tgt = f"cat:{rel['target']}"
            # Weight by relationship type
            w = 0.7 if rel["type"] in ("similar_to", "related_to") else 0.5
            add_edge(src, tgt, w)

        # 2. Position → tension links
        for pos in self.positions.items:
            pos_id = f"pos:{pos.id}"
            for ten_id in pos.related_tensions:
                add_edge(pos_id, f"ten:{ten_id}", 0.8)
            for prec_id in pos.related_precedents:
                add_edge(pos_id, f"pre:{prec_id}", 0.7)
            # Position → its categories
            for cat in pos.categories:
                add_edge(pos_id, f"cat:{cat}", 0.5)

        # 3. Tension adjacency + category links
        for ten in self.tensions.items:
            ten_id = f"ten:{ten.id}"
            for adj_id in ten.adjacent_tensions:
                add_edge(ten_id, f"ten:{adj_id}", 0.6)
            # Tension → related positions (from related_positions dict)
            for pole_positions in ten.related_positions.values():
                if isinstance(pole_positions, list):
                    for pid in pole_positions:
                        add_edge(ten_id, f"pos:{pid}", 0.8)
            for cat in ten.categories:
                add_edge(ten_id, f"cat:{cat}", 0.5)

        # 4. Precedent → dependencies + categories
        for prec in self.precedents.items:
            prec_id = f"pre:{prec.id}"
            for dep_id in prec.dependencies:
                add_edge(prec_id, f"pos:{dep_id}", 0.7)
            for cat in prec.categories:
                add_edge(prec_id, f"cat:{cat}", 0.5)

        # 5. Category → memory links (lightweight — just count-based)
        for i, mem in enumerate(self.memory_system.memories):
            mem_id = f"mem:{i}"
            for cat in mem.get("categories", []):
                add_edge(mem_id, f"cat:{cat}", 0.3)

        return adj

    def _temporal_weight(self, timestamp_str: str, half_life_days: float) -> float:
        """Compute temporal decay weight for a connection."""
        if not timestamp_str:
            return 0.5  # Unknown age — moderate weight
        try:
            created = datetime.fromisoformat(timestamp_str)
            age_seconds = (datetime.now() - created).total_seconds()
            age_days = age_seconds / 86400
            return math.exp(-math.log(2) * age_days / half_life_days)
        except (ValueError, TypeError):
            return 0.5

    def _get_node_info(self, node_id: str) -> Tuple[str, str, dict]:
        """Get (node_type, content, metadata) for a node ID."""
        prefix, obj_id = node_id.split(":", 1)

        if prefix == "pos":
            item = self.positions.get_by_id(obj_id)
            if item:
                return "position", item.statement, {
                    "confidence": item.confidence, "context": item.context,
                    "timestamp": item.timestamp,
                }
        elif prefix == "ten":
            item = self.tensions.get_by_id(obj_id)
            if item:
                return "tension", item.description, {
                    "poles": item.poles, "status": item.status,
                    "timestamp": item.timestamp,
                }
        elif prefix == "pre":
            item = self.precedents.get_by_id(obj_id)
            if item:
                return "precedent", item.statement, {
                    "held": item.held, "context": item.context,
                    "timestamp": item.timestamp,
                }
        elif prefix == "cat":
            cat = self.memory_system.categories.get(obj_id)
            if cat:
                return "category", obj_id, {"description": cat.get("description", "")}
        elif prefix == "mem":
            idx = int(obj_id)
            if 0 <= idx < len(self.memory_system.memories):
                mem = self.memory_system.memories[idx]
                return "memory", mem["content"][:200], {
                    "categories": mem.get("categories", []),
                    "timestamp": mem.get("timestamp", ""),
                }

        return "unknown", "", {}

    def activate(
        self,
        seed_queries: List[str],
        config: ActivationConfig = None,
    ) -> List[ActivatedNode]:
        """
        Propagate activation from seed queries through the network.

        Args:
            seed_queries: Text queries to seed activation from (searched semantically).
            config: Activation parameters.

        Returns:
            List of activated nodes, sorted by activation level descending.
        """
        if config is None:
            config = ActivationConfig()

        adj = self._build_adjacency()

        # Seed activation: find starting nodes from semantic search
        activated: Dict[str, ActivatedNode] = {}
        seed_nodes = self._find_seed_nodes(seed_queries, config)

        for node_id, score in seed_nodes:
            node_type, content, meta = self._get_node_info(node_id)
            activated[node_id] = ActivatedNode(
                node_id=node_id,
                node_type=node_type,
                activation=score,
                depth=0,
                path=[node_id],
                content=content,
                metadata=meta,
            )

        # Multi-hop propagation
        frontier = list(activated.keys())
        for hop in range(1, config.max_hops + 1):
            next_frontier = []

            for src_id in frontier:
                src_activation = activated[src_id].activation
                if src_activation < config.min_activation:
                    continue

                neighbors = adj.get(src_id, [])
                for neighbor_id, edge_weight in neighbors:
                    # Apply temporal decay to the edge
                    _, _, n_meta = self._get_node_info(neighbor_id)
                    temporal = self._temporal_weight(
                        n_meta.get("timestamp", ""), config.temporal_half_life_days
                    )

                    # Propagated activation = source * decay * edge_weight * temporal
                    prop_activation = (
                        src_activation * config.decay_rate * edge_weight * temporal
                    )

                    if prop_activation < config.min_activation:
                        continue

                    if neighbor_id in activated:
                        # Accumulate activation (don't replace if higher already)
                        existing = activated[neighbor_id]
                        if prop_activation > existing.activation:
                            existing.activation = prop_activation
                            existing.depth = min(existing.depth, hop)
                            existing.path = activated[src_id].path + [neighbor_id]
                    else:
                        node_type, content, meta = self._get_node_info(neighbor_id)
                        activated[neighbor_id] = ActivatedNode(
                            node_id=neighbor_id,
                            node_type=node_type,
                            activation=prop_activation,
                            depth=hop,
                            path=activated[src_id].path + [neighbor_id],
                            content=content,
                            metadata=meta,
                        )
                        next_frontier.append(neighbor_id)

            frontier = next_frontier

        # Lateral inhibition: strongly activated nodes suppress weak neighbors
        self._apply_lateral_inhibition(activated, adj, config)

        # Sort by activation
        result = sorted(activated.values(), key=lambda n: n.activation, reverse=True)
        return result

    def _find_seed_nodes(
        self, queries: List[str], config: ActivationConfig
    ) -> List[Tuple[str, float]]:
        """Find seed nodes by semantic search across all types."""
        seeds = []
        seen = set()

        for query in queries:
            # Search positions
            for r in self.positions.search(query, top_k=config.semantic_neighbor_k):
                nid = f"pos:{r['item'].id}"
                if nid not in seen:
                    seeds.append((nid, r["similarity"]))
                    seen.add(nid)

            # Search tensions
            for r in self.tensions.search(query, top_k=config.semantic_neighbor_k):
                nid = f"ten:{r['item'].id}"
                if nid not in seen:
                    seeds.append((nid, r["similarity"]))
                    seen.add(nid)

            # Search precedents
            for r in self.precedents.search(query, top_k=config.semantic_neighbor_k):
                nid = f"pre:{r['item'].id}"
                if nid not in seen:
                    seeds.append((nid, r["similarity"]))
                    seen.add(nid)

            # Search generic memories
            for r in self.memory_system.retrieve_by_similarity(query, top_k=config.semantic_neighbor_k):
                nid = f"mem:{r['memory_id']}"
                if nid not in seen:
                    seeds.append((nid, r["similarity"]))
                    seen.add(nid)

        return seeds

    def _apply_lateral_inhibition(
        self,
        activated: Dict[str, ActivatedNode],
        adj: Dict[str, List[Tuple[str, float]]],
        config: ActivationConfig,
    ):
        """Strong nodes suppress weakly connected neighbors."""
        strong_nodes = [
            n for n in activated.values()
            if n.activation >= config.inhibition_threshold
        ]

        for strong in strong_nodes:
            neighbors = adj.get(strong.node_id, [])
            for neighbor_id, edge_weight in neighbors:
                if neighbor_id in activated:
                    neighbor = activated[neighbor_id]
                    # Only suppress nodes that are weaker and not directly relevant
                    if neighbor.activation < strong.activation * 0.5:
                        neighbor.activation *= (1.0 - config.inhibition_factor)

    def find_adjacent_territory(
        self,
        current_context: str,
        config: ActivationConfig = None,
    ) -> Dict:
        """
        Find unexplored territory bordering the user's intellectual topology.

        Uses activation to find weakly-activated-but-reachable nodes —
        the gaps where the topology has structure nearby but hasn't been explored.

        Returns:
            {
                "explored": [...],      # Strongly activated (known territory)
                "adjacent": [...],      # Weakly activated via propagation (frontier)
                "gap_tensions": [...],  # Tensions near the frontier
            }
        """
        if config is None:
            config = ActivationConfig(max_hops=3)

        nodes = self.activate([current_context], config)

        explored = []
        adjacent = []
        gap_tensions = []

        for node in nodes:
            entry = {
                "id": node.node_id,
                "type": node.node_type,
                "content": node.content[:150],
                "activation": round(node.activation, 3),
                "depth": node.depth,
                "path": node.path,
            }
            entry.update({k: v for k, v in node.metadata.items() if k != "timestamp"})

            if node.depth == 0 or node.activation > 0.4:
                explored.append(entry)
            elif node.depth >= 1 and 0.05 < node.activation <= 0.4:
                adjacent.append(entry)

                # If it's a tension, it's a gap tension
                if node.node_type == "tension":
                    gap_tensions.append(entry)

        return {
            "explored": explored[:10],
            "adjacent": adjacent[:10],
            "gap_tensions": gap_tensions[:5],
        }
