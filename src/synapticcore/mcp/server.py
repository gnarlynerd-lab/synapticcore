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
) -> str:
    """Store structured intellectual content from a conversation.

    Not a transcript or summary — store the positions held, tensions surfaced,
    and precedents established or tested.

    Each position: {"statement": str, "confidence": "tentative"|"held"|"committed", "context": str}
    Each tension: {"poles": [str, str], "description": str, "status": "active"|"explored"|"resolved"}
    Each precedent: {"statement": str, "held": bool, "context": str}
    """
    core = get_core()
    stored = []

    for pos in (positions or []):
        mid = core.memory.add_memory(
            content=pos.get("statement", ""),
            categories=["position"],
            metadata={
                "type": "position",
                "confidence": pos.get("confidence", "tentative"),
                "context": pos.get("context", ""),
            }
        )
        stored.append(f"position:{mid}")

    for ten in (tensions or []):
        poles = ten.get("poles", [])
        mid = core.memory.add_memory(
            content=ten.get("description", " vs ".join(poles)),
            categories=["tension"],
            metadata={
                "type": "tension",
                "poles": poles,
                "status": ten.get("status", "active"),
            }
        )
        stored.append(f"tension:{mid}")

    for prec in (precedents or []):
        mid = core.memory.add_memory(
            content=prec.get("statement", ""),
            categories=["precedent"],
            metadata={
                "type": "precedent",
                "held": prec.get("held", True),
                "context": prec.get("context", ""),
            }
        )
        stored.append(f"precedent:{mid}")

    if session_summary:
        core.memory.add_memory(
            content=session_summary,
            categories=["session_summary"],
            metadata={"type": "session_summary"}
        )

    return f"Stored {len(stored)} items: {', '.join(stored)}"


@mcp.tool()
def retrieve_relevant(
    current_context: str,
    depth: str = "structural",
    max_results: int = 5,
) -> list[dict]:
    """Retrieve intellectually relevant material from the user's history.

    Returns not just semantically similar content but associatively related material.

    Args:
        current_context: What the user is currently exploring.
        depth: "surface" (semantic only), "structural" (+ categories), "deep" (+ associative expansion).
        max_results: Maximum results to return.
    """
    core = get_core()

    depth_to_recursive = {"surface": 0, "structural": 1, "deep": 2}
    recursive_depth = depth_to_recursive.get(depth, 1)

    results = core.memory.enhanced_hybrid_search(
        current_context,
        top_k=max_results,
        recursive_depth=recursive_depth,
    )

    output = []
    for r in results:
        memory = r["memory"]
        entry = {
            "content": memory["content"],
            "categories": memory.get("categories", []),
            "type": memory.get("metadata", {}).get("type", "memory"),
            "score": round(r["combined_score"], 3),
            "retrieval_method": r.get("retrieval_method", "unknown"),
            "timestamp": memory.get("timestamp", ""),
        }
        # Include type-specific metadata
        meta = memory.get("metadata", {})
        if meta.get("type") == "position":
            entry["confidence"] = meta.get("confidence")
            entry["context"] = meta.get("context")
        elif meta.get("type") == "tension":
            entry["poles"] = meta.get("poles")
            entry["status"] = meta.get("status")
        elif meta.get("type") == "precedent":
            entry["held"] = meta.get("held")
            entry["context"] = meta.get("context")
        output.append(entry)

    return output


@mcp.tool()
def get_intellectual_arc(
    topic: str,
    time_range: str | None = None,
) -> dict:
    """Trace how the user's position on a topic has evolved over time.

    Args:
        topic: The subject or tension to trace.
        time_range: Optional date range filter (not yet implemented).

    Returns a chronological sequence of related positions and tensions.
    """
    core = get_core()

    # Search for positions related to this topic
    results = core.memory.enhanced_hybrid_search(
        topic, top_k=20, recursive_depth=1
    )

    # Filter to positions and tensions, sort chronologically
    positions = []
    tensions = []
    for r in results:
        meta = r["memory"].get("metadata", {})
        entry = {
            "content": r["memory"]["content"],
            "timestamp": r["memory"].get("timestamp", ""),
            "score": round(r["combined_score"], 3),
        }
        if meta.get("type") == "position":
            entry["confidence"] = meta.get("confidence")
            entry["context"] = meta.get("context")
            positions.append(entry)
        elif meta.get("type") == "tension":
            entry["poles"] = meta.get("poles")
            entry["status"] = meta.get("status")
            tensions.append(entry)

    # Sort chronologically
    positions.sort(key=lambda x: x["timestamp"])
    tensions.sort(key=lambda x: x["timestamp"])

    # Also include general memories that are highly relevant
    general = [
        {
            "content": r["memory"]["content"],
            "timestamp": r["memory"].get("timestamp", ""),
            "categories": r["memory"].get("categories", []),
            "score": round(r["combined_score"], 3),
        }
        for r in results
        if r["memory"].get("metadata", {}).get("type") not in ("position", "tension", "precedent")
        and r["combined_score"] > 0.3
    ]
    general.sort(key=lambda x: x["timestamp"])

    return {
        "topic": topic,
        "positions": positions,
        "tensions": tensions,
        "related_memories": general[:5],
        "note": "v1: chronological ordering only. Arc detection (reinforcement/shift/reversal) coming in Phase 4."
    }


@mcp.tool()
def find_adjacent(
    current_context: str,
) -> dict:
    """Identify unexplored territory bordering the user's intellectual topology.

    Finds tensions that haven't been explored, categories that connect to
    the current conversation but haven't been tested, and gaps in the topology.
    """
    core = get_core()

    # Get what's directly relevant
    direct_results = core.memory.enhanced_hybrid_search(
        current_context, top_k=5, recursive_depth=2
    )

    # Get categories touched by direct results
    touched_categories = set()
    for r in direct_results:
        touched_categories.update(r["memory"].get("categories", []))

    # Find related categories through the relationship graph
    related_categories = []
    if core.enhanced:
        rels = core.enhanced.discover_category_relationships(min_similarity=0.4)
        for rel in rels:
            if rel["source"] in touched_categories and rel["target"] not in touched_categories:
                related_categories.append({"category": rel["target"], "similarity": round(rel["similarity"], 3),
                                            "connected_via": rel["source"]})
            elif rel["target"] in touched_categories and rel["source"] not in touched_categories:
                related_categories.append({"category": rel["source"], "similarity": round(rel["similarity"], 3),
                                            "connected_via": rel["target"]})

    # Find uncategorized clusters that might be adjacent
    new_category_suggestions = []
    if core.enhanced:
        clusters = core.enhanced.suggest_new_categories(min_memories=2, similarity_threshold=0.5)
        for cluster in clusters:
            new_category_suggestions.append({
                "size": cluster["size"],
                "sample_texts": cluster["sample_texts"],
            })

    return {
        "current_territory": list(touched_categories),
        "adjacent_categories": related_categories[:5],
        "unexplored_clusters": new_category_suggestions[:3],
        "note": "v1: category-based adjacency. Topology-aware adjacency coming in Phase 5."
    }


@mcp.tool()
def check_precedent(
    context: str,
) -> list[dict]:
    """Look up whether the user has established relevant precedents.

    Searches for positions they committed to previously, whether those
    precedents held or broke, and implications for the current conversation.
    """
    core = get_core()

    # Search specifically for precedents
    results = core.memory.enhanced_hybrid_search(
        context, top_k=10, recursive_depth=1
    )

    precedents = []
    for r in results:
        meta = r["memory"].get("metadata", {})
        if meta.get("type") == "precedent":
            precedents.append({
                "statement": r["memory"]["content"],
                "held": meta.get("held", True),
                "context": meta.get("context", ""),
                "timestamp": r["memory"].get("timestamp", ""),
                "relevance": round(r["combined_score"], 3),
            })

    # Also find committed positions (strong precedent-like)
    for r in results:
        meta = r["memory"].get("metadata", {})
        if meta.get("type") == "position" and meta.get("confidence") == "committed":
            precedents.append({
                "statement": r["memory"]["content"],
                "held": True,
                "context": meta.get("context", ""),
                "timestamp": r["memory"].get("timestamp", ""),
                "relevance": round(r["combined_score"], 3),
                "source": "committed_position",
            })

    precedents.sort(key=lambda x: x["relevance"], reverse=True)
    return precedents[:5]


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

    # Search for what connects to this topic
    results = core.memory.enhanced_hybrid_search(
        current_topic, top_k=10, recursive_depth=1
    )

    # Analyze what's available
    categories_touched = set()
    types_found = {"position": 0, "tension": 0, "precedent": 0, "memory": 0}
    high_relevance = []

    for r in results:
        categories_touched.update(r["memory"].get("categories", []))
        mem_type = r["memory"].get("metadata", {}).get("type", "memory")
        if mem_type in types_found:
            types_found[mem_type] += 1
        else:
            types_found["memory"] += 1
        if r["combined_score"] > 0.5:
            high_relevance.append({
                "content": r["memory"]["content"][:100],
                "type": mem_type,
                "score": round(r["combined_score"], 3),
            })

    # Check if current exchange connects to known tensions
    tension_connections = [
        r for r in results
        if r["memory"].get("metadata", {}).get("type") == "tension"
        and r["combined_score"] > 0.3
    ]

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
        "active_tensions_nearby": len(tension_connections),
        "highly_relevant_items": high_relevance[:3],
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
