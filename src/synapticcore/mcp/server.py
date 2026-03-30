"""
SynapticCore MCP Server.

Exposes user narrative memory tools via the Model Context Protocol.
Tracks decisions, tradeoffs, and constraints — not atomic facts.
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
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from mcp.server.fastmcp import FastMCP

from .. import SynapticCore

# ── Server setup ─────────────────────────────────────────────────────────

STORAGE_PATH = os.environ.get("SYNAPTICCORE_STORAGE", "memory_store.json")

mcp = FastMCP(
    name="SynapticCore",
    instructions=(
        "SynapticCore tracks how a user navigates decisions, tradeoffs, and "
        "constraints over time. It doesn't store atomic facts — it maintains "
        "the structured map of what's settled, what's still open, and what "
        "constrains what. "
        "Use store_interaction after meaningful exchanges to record decisions, "
        "tradeoffs, and constraints. "
        "Use retrieve_relevant to get context before responding — in deep mode "
        "it returns structurally adjacent material through spreading activation. "
        "Use get_decision_narrative to trace how the user's approach evolved. "
        "Use find_adjacent to discover unaddressed territory. "
        "Use check_constraints to find constraints that should apply. "
        "Use assess_depth to check if deeper context is available."
    ),
)

_core: SynapticCore | None = None


def get_core() -> SynapticCore:
    global _core
    if _core is None:
        _core = SynapticCore(storage_path=STORAGE_PATH)
    return _core


# ── Tools ────────────────────────────────────────────────────────────────


@mcp.tool()
def store_interaction(
    decisions: list[dict] | None = None,
    tradeoffs: list[dict] | None = None,
    constraints: list[dict] | None = None,
    session_summary: str = "",
    relationship_hint: str = "",
) -> str:
    """Store decisions, tradeoffs, and constraints from a conversation.

    Not a transcript or summary — store what was decided, what's still
    unresolved, and what constrains future choices.

    Each decision: {"statement": str, "confidence": "tentative"|"held"|"committed", "context": str}
      - tentative: exploring, not settled
      - held: current working position, could change
      - committed: settled, should bind future actions

    Each tradeoff: {"poles": [str, str], "description": str, "status": "active"|"explored"|"resolved"}
      - Recurring tradeoffs are features, not bugs. Track them.

    Each constraint: {"statement": str, "held": bool, "context": str}
      - A decision that was made and should bind future actions unless explicitly revisited.

    relationship_hint: Optional hint about how new decisions relate to existing ones,
        e.g. "contradicts previous decision on X" or "reinforces the budget commitment".
        The calling LLM has conversational context the memory system doesn't.
    """
    core = get_core()
    stored = []
    arc_detections = []

    for dec in (decisions or []):
        obj = core.positions.create(
            statement=dec.get("statement", ""),
            confidence=dec.get("confidence", "tentative"),
            context=dec.get("context", ""),
        )
        stored.append(f"decision:{obj.id[:8]}")
        detection = core.arcs.detect_relationship(obj, relationship_hint=relationship_hint)
        if detection:
            arc_detections.append(detection)

    for trd in (tradeoffs or []):
        obj = core.tensions.create(
            poles=trd.get("poles", []),
            description=trd.get("description", ""),
            status=trd.get("status", "active"),
        )
        stored.append(f"tradeoff:{obj.id[:8]}")

    for con in (constraints or []):
        obj = core.precedents.create(
            statement=con.get("statement", ""),
            held=con.get("held", True),
            context=con.get("context", ""),
        )
        stored.append(f"constraint:{obj.id[:8]}")

    if session_summary:
        core.memory.add_memory(
            content=session_summary,
            categories=["session_summary"],
            metadata={"type": "session_summary"}
        )

    core.save()

    parts = [f"Stored {len(stored)} items: {', '.join(stored)}"]
    for det in arc_detections:
        parts.append(f"Narrative arc: {det['type']} — {det['explanation']}")

    return "\n".join(parts)


def _typed_search_results(core, query: str, top_k: int = 5) -> list[dict]:
    """Search across all typed managers and return unified results."""
    output = []
    for manager, type_name in [
        (core.positions, "decision"),
        (core.tensions, "tradeoff"),
        (core.precedents, "constraint"),
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
            if type_name == "decision":
                entry["content"] = item.statement
                entry["confidence"] = item.confidence
                entry["context"] = item.context
                entry["evolution"] = item.evolution
                entry["categories"] = item.categories
            elif type_name == "tradeoff":
                entry["content"] = item.description
                entry["poles"] = item.poles
                entry["status"] = item.status
                entry["engagement_count"] = len(item.engagement_history)
                entry["categories"] = item.categories
            elif type_name == "constraint":
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
    """Retrieve relevant context from the user's decision landscape.

    Returns decisions, tradeoffs, and constraints that connect to the
    current conversation — not just semantically similar content but
    structurally adjacent material from the user's narrative.

    Args:
        current_context: What the user is currently working on or discussing.
        depth: "surface" (semantic only), "structural" (+ typed search),
               "deep" (+ spreading activation through the decision landscape).
        max_results: Maximum results to return.
    """
    core = get_core()

    if depth == "deep":
        from ..retrieval.activation import ActivationConfig
        config = ActivationConfig(max_hops=3, decay_rate=0.5)
        nodes = core.activation.activate([current_context], config)

        # Map internal types to user-facing vocabulary
        type_map = {"position": "decision", "tension": "tradeoff", "precedent": "constraint"}

        output = []
        for node in nodes:
            if node.node_type == "category":
                continue
            entry = {
                "content": node.content[:300],
                "type": type_map.get(node.node_type, node.node_type),
                "score": round(node.activation, 3),
                "retrieval_method": f"activation_depth_{node.depth}",
                "activation_path": [p.split(":", 1)[-1][:12] for p in node.path],
                "timestamp": node.metadata.get("timestamp", ""),
            }
            entry.update({k: v for k, v in node.metadata.items()
                          if k not in ("timestamp",)})
            output.append(entry)
        return output[:max_results]

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
def get_decision_narrative(
    topic: str,
    time_range: str | None = None,
) -> dict:
    """Trace how the user's approach to a topic evolved over time.

    Returns a chronological chain of decisions with transition classifications
    (reinforcement/shift/reversal), trajectory analysis (converging/oscillating/
    deepening), and recurring tradeoffs.

    Args:
        topic: The subject to trace (e.g. "kitchen budget", "architecture choices").
        time_range: Optional date range filter (not yet implemented).
    """
    core = get_core()
    arc = core.arcs.get_arc(topic, time_range=time_range)

    # Translate vocabulary in the response
    return {
        "topic": arc["topic"],
        "decisions": arc["positions"],  # internal "positions" → user-facing "decisions"
        "trajectory": arc["trajectory"],
        "recurring_tradeoffs": arc["recurring_tensions"],
        "summary": arc["summary"],
    }


@mcp.tool()
def find_adjacent(
    current_context: str,
) -> dict:
    """Identify unaddressed territory bordering the current conversation.

    Uses spreading activation through the decision landscape to find
    tradeoffs that haven't been explored, decisions that connect to the
    current context but haven't been made, and gaps where constraints
    may apply but haven't been checked.
    """
    core = get_core()
    result = core.activation.find_adjacent_territory(current_context)

    # Map internal types in results
    type_map = {"position": "decision", "tension": "tradeoff", "precedent": "constraint"}
    for section in ("explored", "adjacent", "gap_tensions"):
        for item in result.get(section, []):
            if "type" in item:
                item["type"] = type_map.get(item["type"], item["type"])

    return result


@mcp.tool()
def check_constraints(
    context: str,
) -> list[dict]:
    """Find constraints that should apply to the current situation.

    Returns constraints with test histories showing whether they've held
    or been violated. Also returns committed decisions that function as
    binding constraints.
    """
    core = get_core()
    output = []

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
            "source": "constraint",
        })

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
                "source": "committed_decision",
            })

    output.sort(key=lambda x: x["relevance"], reverse=True)
    return output[:5]


@mcp.tool()
def assess_depth(
    recent_exchange: str,
    current_topic: str,
) -> dict:
    """Check if the conversation is engaging with the user's decision landscape.

    Information tool — tells the LLM "there's relevant context here if you
    want to go deeper." Not a judgment.
    """
    core = get_core()

    pos_results = core.positions.search(current_topic, top_k=5)
    ten_results = core.tensions.search(current_topic, top_k=5)
    prec_results = core.precedents.search(current_topic, top_k=5)

    material = {
        "decisions": len(pos_results),
        "tradeoffs": len(ten_results),
        "constraints": len(prec_results),
    }

    categories_touched = set()
    for r in pos_results:
        categories_touched.update(r["item"].categories)
    for r in ten_results:
        categories_touched.update(r["item"].categories)

    high_relevance = []
    for r in pos_results[:2]:
        high_relevance.append({
            "content": r["item"].statement[:100],
            "type": "decision",
            "confidence": r["item"].confidence,
            "score": round(r["similarity"], 3),
        })
    for r in ten_results[:2]:
        high_relevance.append({
            "content": r["item"].description[:100],
            "type": "tradeoff",
            "poles": r["item"].poles,
            "score": round(r["similarity"], 3),
        })

    active_tradeoffs = [r for r in ten_results if r["item"].status == "active"]
    all_recurring = core.arcs.detect_recurring_tensions(min_engagements=2)
    recurring_nearby = [t for t in all_recurring if any(
        r["item"].id == t["id"] for r in ten_results
    )]

    has_deeper_structure = any(v > 0 for v in material.values())

    return {
        "topic": current_topic,
        "categories_connected": list(categories_touched),
        "narrative_material": material,
        "has_deeper_structure": has_deeper_structure,
        "active_tradeoffs_nearby": len(active_tradeoffs),
        "recurring_tradeoffs_nearby": len(recurring_nearby),
        "highly_relevant_items": high_relevance,
        "suggestion": (
            "There are decisions, tradeoffs, or constraints that connect to this topic. "
            "Consider checking them before proceeding."
            if has_deeper_structure
            else "This appears to be new territory. Consider storing decisions or tradeoffs that emerge."
        ),
    }


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
