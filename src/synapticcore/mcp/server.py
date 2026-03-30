"""
SynapticCore MCP Server.

Exposes cognitive memory tools via the Model Context Protocol.
Run with: python -m synapticcore.mcp.server
"""

import logging
import os
import sys

# Route all logging to stderr so stdout stays clean for MCP JSON protocol
logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="%(name)s: %(message)s",
)
# Suppress noisy libraries
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# Suppress HuggingFace progress bars
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from mcp.server.fastmcp import FastMCP

from .. import SynapticCore

# ── Server setup ─────────────────────────────────────────────────────────

STORAGE_PATH = os.environ.get("SYNAPTICCORE_STORAGE", "memory_store.json")

mcp = FastMCP(
    name="SynapticCore",
    instructions=(
        "SynapticCore is a cognitive memory system that tracks how your thinking evolves. "
        "It stores positions, tensions, and precedents — not atomic facts — and retrieves "
        "structurally adjacent material through spreading activation. "
        "Use store_interaction after meaningful exchanges. "
        "Use retrieve_relevant to get context before responding. "
        "Use get_intellectual_arc to trace how positions evolved. "
        "Use find_adjacent to discover unexplored territory. "
        "Use check_precedent to look up established commitments. "
        "Use assess_depth to evaluate conversation engagement with deeper structure."
    ),
)

# Initialize core on first use (lazy to avoid slow startup blocking MCP handshake)
_core: SynapticCore | None = None


def get_core() -> SynapticCore:
    global _core
    if _core is None:
        _core = SynapticCore(storage_path=STORAGE_PATH)
    return _core


# ── Tools ────────────────────────────────────────────────────────────────


@mcp.tool()
def store_interaction(
    positions: list[dict] | None = None,
    tensions: list[dict] | None = None,
    precedents: list[dict] | None = None,
    session_summary: str = "",
    relationship_hint: str = "",
) -> str:
    """Store structured intellectual content from a conversation.

    Not a transcript or summary — store the positions held, tensions surfaced,
    and precedents established or tested.

    Each position: {"statement": str, "confidence": "tentative"|"held"|"committed", "context": str}
    Each tension: {"poles": [str, str], "description": str, "status": "active"|"explored"|"resolved"}
    Each precedent: {"statement": str, "held": bool, "context": str}
    relationship_hint: Optional hint about how new positions relate to existing ones,
        e.g. "contradicts previous position on X" or "reinforces earlier commitment to Y".
        The calling LLM has conversational context the memory system doesn't.
    """
    core = get_core()
    stored = []
    arc_detections = []

    for pos in (positions or []):
        obj = core.positions.create(
            statement=pos.get("statement", ""),
            confidence=pos.get("confidence", "tentative"),
            context=pos.get("context", ""),
        )
        stored.append(f"position:{obj.id[:8]}")

        # Auto-detect relationship to existing positions
        detection = core.arcs.detect_relationship(obj, relationship_hint=relationship_hint)
        if detection:
            arc_detections.append(detection)

    for ten in (tensions or []):
        obj = core.tensions.create(
            poles=ten.get("poles", []),
            description=ten.get("description", ""),
            status=ten.get("status", "active"),
        )
        stored.append(f"tension:{obj.id[:8]}")

    for prec in (precedents or []):
        obj = core.precedents.create(
            statement=prec.get("statement", ""),
            held=prec.get("held", True),
            context=prec.get("context", ""),
        )
        stored.append(f"precedent:{obj.id[:8]}")

    if session_summary:
        core.memory.add_memory(
            content=session_summary,
            categories=["session_summary"],
            metadata={"type": "session_summary"}
        )

    # Persist typed data
    core.save()

    # Build response
    parts = [f"Stored {len(stored)} items: {', '.join(stored)}"]
    for det in arc_detections:
        parts.append(
            f"Arc detected: {det['type']} — {det['explanation']}"
        )

    return "\n".join(parts)


def _typed_search_results(core, query: str, top_k: int = 5) -> list[dict]:
    """Search across all typed managers and return unified results."""
    output = []
    for manager, type_name in [
        (core.positions, "position"),
        (core.tensions, "tension"),
        (core.precedents, "precedent"),
    ]:
        for r in manager.search(query, top_k=top_k):
            item = r["item"]
            entry = {
                "type": type_name,
                "id": item.id,
                "score": round(r["similarity"], 3),
                "timestamp": item.timestamp,
                "retrieval_method": "typed_semantic",
            }
            if type_name == "position":
                entry["content"] = item.statement
                entry["confidence"] = item.confidence
                entry["context"] = item.context
                entry["evolution"] = item.evolution
                entry["categories"] = item.categories
            elif type_name == "tension":
                entry["content"] = item.description
                entry["poles"] = item.poles
                entry["status"] = item.status
                entry["engagement_count"] = len(item.engagement_history)
                entry["categories"] = item.categories
            elif type_name == "precedent":
                entry["content"] = item.statement
                entry["held"] = item.held
                entry["context"] = item.context
                entry["test_count"] = len(item.test_history)
                entry["categories"] = item.categories
            output.append(entry)
    return output


@mcp.tool()
def retrieve_relevant(
    current_context: str,
    depth: str = "structural",
    max_results: int = 5,
) -> list[dict]:
    """Retrieve intellectually relevant material from the user's history.

    Returns not just semantically similar content but structurally adjacent
    material — things connected through the intellectual topology, not just
    embedding similarity.

    Args:
        current_context: What the user is currently exploring.
        depth: "surface" (semantic only), "structural" (+ typed search),
               "deep" (+ spreading activation through topology).
        max_results: Maximum results to return.
    """
    core = get_core()

    if depth == "deep":
        # Use spreading activation for deep retrieval
        from ..retrieval.activation import ActivationConfig
        config = ActivationConfig(max_hops=3, decay_rate=0.5)
        nodes = core.activation.activate([current_context], config)

        output = []
        for node in nodes:
            if node.node_type == "category":
                continue  # Skip category nodes in output
            entry = {
                "content": node.content[:300],
                "type": node.node_type,
                "score": round(node.activation, 3),
                "retrieval_method": f"activation_depth_{node.depth}",
                "activation_path": [p.split(":", 1)[-1][:12] for p in node.path],
                "timestamp": node.metadata.get("timestamp", ""),
            }
            entry.update({k: v for k, v in node.metadata.items()
                          if k not in ("timestamp",)})
            output.append(entry)
        return output[:max_results]

    # Surface and structural: semantic + typed search
    depth_to_recursive = {"surface": 0, "structural": 1}
    recursive_depth = depth_to_recursive.get(depth, 1)

    memory_results = core.memory.enhanced_hybrid_search(
        current_context, top_k=max_results, recursive_depth=recursive_depth,
    )
    output = []
    for r in memory_results:
        memory = r["memory"]
        output.append({
            "content": memory["content"],
            "categories": memory.get("categories", []),
            "type": memory.get("metadata", {}).get("type", "memory"),
            "score": round(r["combined_score"], 3),
            "retrieval_method": r.get("retrieval_method", "unknown"),
            "timestamp": memory.get("timestamp", ""),
        })

    typed_results = _typed_search_results(core, current_context, top_k=max_results)
    output.extend(typed_results)

    output.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    deduped = []
    for item in output:
        key = item.get("content", "")[:100]
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:max_results]


@mcp.tool()
def get_intellectual_arc(
    topic: str,
    time_range: str | None = None,
) -> dict:
    """Trace how the user's position on a topic has evolved over time.

    Returns a chronological chain of positions with transition classifications
    (reinforcement/shift/reversal), trajectory analysis (converging/oscillating/
    deepening), and recurring tensions.

    Args:
        topic: The subject or tension to trace.
        time_range: Optional date range filter (not yet implemented).
    """
    core = get_core()
    return core.arcs.get_arc(topic, time_range=time_range)


@mcp.tool()
def find_adjacent(
    current_context: str,
) -> dict:
    """Identify unexplored territory bordering the user's intellectual topology.

    Uses spreading activation to find weakly-activated-but-reachable nodes —
    territory that is structurally close to your current thinking but hasn't
    been explored. Returns the frontier of your intellectual topology.
    """
    core = get_core()
    return core.activation.find_adjacent_territory(current_context)


@mcp.tool()
def check_precedent(
    context: str,
) -> list[dict]:
    """Look up whether the user has established relevant precedents.

    Searches for precedents and committed positions. Returns whether
    they held or broke under pressure, with test history.
    """
    core = get_core()
    output = []

    # Search typed precedents
    for r in core.precedents.search(context, top_k=5):
        p = r["item"]
        output.append({
            "id": p.id,
            "statement": p.statement,
            "held": p.held,
            "context": p.context,
            "timestamp": p.timestamp,
            "test_history": p.test_history,
            "dependencies": p.dependencies,
            "relevance": round(r["similarity"], 3),
            "source": "precedent",
        })

    # Also find committed positions (precedent-like)
    for r in core.positions.search(context, top_k=5):
        p = r["item"]
        if p.confidence == "committed":
            output.append({
                "id": p.id,
                "statement": p.statement,
                "held": True,
                "context": p.context,
                "timestamp": p.timestamp,
                "test_history": [],
                "relevance": round(r["similarity"], 3),
                "source": "committed_position",
            })

    output.sort(key=lambda x: x["relevance"], reverse=True)
    return output[:5]


@mcp.tool()
def assess_depth(
    recent_exchange: str,
    current_topic: str,
) -> dict:
    """Evaluate whether the conversation engages with the user's intellectual topology.

    This is an information tool — it tells the LLM "there's more here if you
    want to go deeper" rather than making judgments.
    """
    core = get_core()

    # Count typed objects related to this topic
    pos_results = core.positions.search(current_topic, top_k=5)
    ten_results = core.tensions.search(current_topic, top_k=5)
    prec_results = core.precedents.search(current_topic, top_k=5)

    types_found = {
        "position": len(pos_results),
        "tension": len(ten_results),
        "precedent": len(prec_results),
    }

    # Collect categories across all typed results
    categories_touched = set()
    for r in pos_results:
        categories_touched.update(r["item"].categories)
    for r in ten_results:
        categories_touched.update(r["item"].categories)

    # High relevance items
    high_relevance = []
    for r in pos_results[:2]:
        high_relevance.append({
            "content": r["item"].statement[:100],
            "type": "position",
            "confidence": r["item"].confidence,
            "score": round(r["similarity"], 3),
        })
    for r in ten_results[:2]:
        high_relevance.append({
            "content": r["item"].description[:100],
            "type": "tension",
            "poles": r["item"].poles,
            "score": round(r["similarity"], 3),
        })

    # Check for active tensions
    active_tensions = [r for r in ten_results if r["item"].status == "active"]

    # Check recurring tensions via arc tracker
    all_recurring = core.arcs.detect_recurring_tensions(min_engagements=2)
    recurring_nearby = [t for t in all_recurring if any(
        r["item"].id == t["id"] for r in ten_results
    )]

    has_deeper_structure = (
        types_found["position"] > 0 or
        types_found["tension"] > 0 or
        types_found["precedent"] > 0
    )

    return {
        "topic": current_topic,
        "categories_connected": list(categories_touched),
        "intellectual_material": types_found,
        "has_deeper_structure": has_deeper_structure,
        "active_tensions_nearby": len(active_tensions),
        "recurring_tensions_nearby": len(recurring_nearby),
        "highly_relevant_items": high_relevance,
        "suggestion": (
            "Deeper intellectual structure is available — positions, tensions, or precedents "
            "connect to this topic. Consider engaging with them."
            if has_deeper_structure
            else "This appears to be new territory. Consider storing positions or tensions that emerge."
        ),
    }


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
