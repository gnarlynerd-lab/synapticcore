# SynapticCore MCP Server — Claude Code Handoff Brief

---

## What This Is

SynapticCore is a cognitive memory architecture exposed as an MCP (Model Context Protocol) server. It gives any MCP-compatible LLM client (Claude, ChatGPT, etc.) the ability to track how a user's thinking evolves over time — not facts and preferences, but intellectual arcs, recurring tensions, positions held and broken, and the topology of commitments that emerges across conversations.

Every existing AI memory system — Claude's native memory, ChatGPT's memory, Mem0, Letta, Zep — stores atomic facts: "user prefers Python," "user works at company X." When a fact changes, the old one is overwritten. SynapticCore stores something fundamentally different: the *structure of engagement*. How positions relate to each other. What tensions keep recurring. What's adjacent to what in someone's reasoning. How commitments evolve rather than simply updating.

The result is a companion that knows you not the way a database knows you, but the way a close colleague who's been thinking alongside you for years knows you. The warmth is in the quality of attention — recognizing connections you don't see, remembering what you said three months ago and noticing it matters now, knowing when you're retreating to a comfortable position versus genuinely exploring.

---

## What Exists — Current State

See `PROJECT_STATUS.md` for the full file map and architecture diagram. Summary of what's working and what isn't:

### Working Solid
- **Core memory system** (`simple_memory_system.py`, ~1277 lines) — full CRUD, UUID-based, JSON persistence, sentence-transformer embeddings, HNSW vector indexing
- **4-phase hybrid search** — semantic similarity (KNN), category matching, recency boost (exponential decay), and associative expansion (spreading activation already exists as Phase 4)
- **Dual embedding spaces** (`enhanced_memory_system.py`, ~478 lines) — category-level aggregated embeddings, cross-category similarity, outlier detection, new category suggestions via clustering
- **Category management** — dynamic creation, version history, deprecation workflow, relationship mapping between categories
- **Feedback loop** (`memory_feedback_loop.py`) — search pattern tracking, failed search logging, category coherence metrics, merge/split/rename suggestions, auto-apply
- **Narrative system** (`narrative_memory_system.py`, ~1635 lines) — narrative creation from memories, confidence-scored points, temporal ordering, narrative search. This is the foundation for intellectual arc tracking.
- **FastAPI wrapper** (`memory_system_api.py`, ~898 lines) — full REST API for all memory operations
- **CLI chat** (`chat_with_memory.py`) — interactive mode with memory-integrated conversation

### Partial or Broken
- **LLM integration** — DeepSeek API calls are wired up but auth is failing (401 errors). Narrative generation and LLM-driven category suggestions depend on this.
- **Web UI** — `memory.html` exists but is minimal
- **HNSW index persistence** — index rebuilds from scratch on every startup

### Doesn't Exist Yet
- **MCP server protocol**
- **Position/tension/precedent as first-class memory types** (the narrative system has infrastructure for temporal tracking but not these specific types)
- **Intellectual arc detection** (position evolution, shift/reinforcement/reversal classification)
- **Adjacency mapping** (unexplored territory bordering the user's topology)
- **Depth assessment** (evaluating whether conversation engages structural patterns)
- **Tests** — zero test files
- **Scalable storage** — everything is single-file JSON; no database backend
- **Multi-user isolation**

### Existing Data
Several JSON stores with real data from development/testing:
- `memory_store.json` (~623 KB, ~19K lines)
- `memory_store_v3.json` (~203 KB)
- `enhanced_memory.json` (~72 KB)
- `conversation_memory.json` (~43 KB)
- `feedback_history.json` (~14 KB)

---

## What Claude Code Needs to Build

### 1. MCP Server Layer

Wrap SynapticCore as an MCP server using the Model Context Protocol specification. The server exposes tools that an LLM client can call during conversation.

Use the official MCP Python SDK: `mcp` (pip installable). The server should run as a standalone process that MCP clients connect to.

**Reference**: https://modelcontextprotocol.io/docs — read the spec for tool definition format, server lifecycle, and transport options. Support both stdio and SSE transports.

### 2. Core Tools to Expose

These are the MCP tools the server makes available to LLM clients:

#### `store_interaction`
After a meaningful exchange, store structured intellectual content — not a transcript, not a summary, but the *positions*, *tensions*, and *precedents* that emerged.

Input:
```json
{
  "positions": [
    {
      "statement": "what the user held",
      "confidence": "how firmly (tentative/held/committed)",
      "context": "what prompted this position"
    }
  ],
  "tensions": [
    {
      "poles": ["pole A", "pole B"],
      "description": "the unresolved opposition",
      "status": "active/explored/resolved"
    }
  ],
  "precedents": [
    {
      "statement": "what was established",
      "held": true,
      "context": "under what pressure it was tested"
    }
  ],
  "session_summary": "one sentence on where the conversation arrived"
}
```

The system should embed these using the dual embedding spaces — content embedding for semantic retrieval, category embedding for structural positioning within the user's intellectual topology.

#### `retrieve_relevant`
Given a current conversational context, return intellectually relevant material from the user's history. This is where SynapticCore's architecture matters most — it should return not just semantically similar content but *associatively related* material that connects through structural adjacency, not just topic similarity.

Input:
```json
{
  "current_context": "what the user is currently exploring",
  "depth": "surface | structural | deep",
  "max_results": 5
}
```

Output should include: relevant positions (with evolution history if they've changed), active tensions that connect to the current context, established precedents that bear on it, and suggested adjacencies.

#### `get_intellectual_arc`
For a given topic or tension, return how the user's position has evolved over time.

Input:
```json
{
  "topic": "the subject or tension to trace",
  "time_range": "optional date range"
}
```

Output: a temporal sequence of positions, what prompted shifts, which precedents held and which broke, and whether the arc shows convergence, oscillation, or deepening.

#### `find_adjacent`
Given the current conversation, identify unexplored territory that borders the user's existing intellectual topology.

Input:
```json
{
  "current_context": "what the user is currently exploring"
}
```

Output: tensions that are structurally adjacent to the current conversation but haven't been explored, positions that connect to current ones through the category relationship graph but haven't been tested, and areas where the topology has gaps or under-explored regions.

#### `check_precedent`
Look up whether the user has established relevant precedents — positions they committed to previously.

Input:
```json
{
  "context": "current situation or position being explored"
}
```

Output: relevant precedents, whether they held or broke, and what the implications are for the current conversation.

#### `assess_depth`
Evaluate whether the current conversation is engaging with the user's actual intellectual topology or staying on the surface. This is not a judgment tool — it's an information tool that helps the LLM understand whether there's deeper structure available.

Input:
```json
{
  "recent_exchange": "last few turns of conversation",
  "current_topic": "what's being discussed"
}
```

Output: whether the conversation connects to known tensions, whether positions are being tested or just stated, and what deeper structures are available nearby.

### 3. New Memory Types

Extend the existing memory system with three new first-class memory types that sit alongside the existing generic memories:

**Positions** — What the user holds or has held. Each position has:
- Statement
- Confidence level (tentative → held → committed)
- Timestamp and context of establishment
- Evolution history (if it shifted, when and why)
- Links to related tensions and precedents

**Tensions** — Recurring unresolved oppositions. Each tension has:
- Two poles
- History of engagement (when it surfaced, how it was explored)
- Status (active/explored/dormant/resolved)
- Adjacent tensions
- Related positions on each side

**Precedents** — Established commitments. Each precedent has:
- Statement of what was established
- Whether it has held under subsequent pressure
- History of testing (when it was challenged, outcome)
- Dependencies (what other positions depend on this precedent)

These should integrate with the existing dual embedding system. Each type gets embedded in both content and category spaces. The category space is where structural relationships live — tensions cluster with related tensions, positions connect to the precedents that support them, etc.

### 4. Intellectual Arc Tracking

This is the novel capability. Build a system that detects and tracks how positions evolve over time:

- When a new position is stored that relates to an existing one, detect whether it's a **reinforcement** (same position, stronger), a **shift** (related but different), or a **reversal** (contradicts the previous position)
- Maintain temporal chains: position A (Jan) → position B (Mar) → position C (Jun)
- Detect **recurring tensions** — tensions that keep surfacing across conversations. These are signals, not errors. The system should flag them as important structural features of the user's thinking
- Identify **convergence** (positions stabilizing over time) and **oscillation** (positions that keep flipping)
- The narrative memory system already has infrastructure for temporal/causal relationships — extend it to handle position evolution specifically

### 5. Enhance Spreading Activation

Phase 4 of the existing hybrid search already implements associative expansion — this is a form of spreading activation. Enhance it with:

- **Lateral inhibition** — strongly activated nodes should suppress weakly related alternatives, preventing retrieval from becoming too diffuse
- **Temporal decay** — activation strength should decay based on how long ago connections were established or last activated
- **Multi-hop propagation** — the existing `recursive_depth` parameter in `enhanced_hybrid_search` is a starting point, but implement proper spreading activation dynamics where activation propagates through the category relationship graph with configurable decay rates
- **Position/tension graph integration** — the new memory types (positions, tensions, precedents) should participate in the activation network. A tension should activate its poles, adjacent tensions, and positions that relate to it

The goal: when the user is talking about X, the system surfaces not just things semantically similar to X, but things that are *structurally adjacent* in their intellectual topology — related tensions, positions that bear on this area, precedents that connect. The "stairs evokes slip" pattern.

### 6. Companionate Intelligence

The system should enable — not enforce — a companionate interaction style. This means:

- When the LLM retrieves context via `retrieve_relevant`, the output should include enough temporal and relational information that the LLM can reference shared history naturally ("this connects to something you were working through in March")
- The `find_adjacent` tool should surface unexplored territory in a way that's inviting, not evaluative
- The `assess_depth` tool provides information without judgment — it tells the LLM "there's more here if you want to go deeper" rather than "the conversation is too shallow"
- Position evolution history should be surfaced with care — "you've held this before and it broke" is useful information, not an accusation

The architecture doesn't dictate disposition. A Socratic protocol can use these tools adversarially. A coaching protocol can use them supportively. A research protocol can use them analytically. The memory layer serves all of them.

---

## Technical Stack

**Current (keep)**:
- **Python 3.8+** with venv at `scagent_env/`
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for embeddings
- **HNSWLib** for vector indexing / approximate nearest neighbor search
- **JSON files** for persistence (existing pattern)

**Add**:
- **MCP SDK**: `mcp` Python package for server implementation
- **Tests**: pytest — the core search and category logic is complex enough to require tests before extending

**Swap or fix**:
- **LLM integration**: The existing DeepSeek integration has auth errors. The MCP server itself doesn't need an LLM for most operations — the calling LLM client handles conversation. But some operations (intellectual arc detection, narrative synthesis, category evolution suggestions) benefit from LLM-driven analysis. Options: fix DeepSeek auth, swap to Anthropic API, or make the LLM provider configurable. Recommend making it configurable with a provider interface.
- **FastAPI**: Keep the existing API as an optional inspection/debugging tool. The primary interface is MCP.

**Defer**:
- PostgreSQL + pgvector (later, when JSON files hit scale limits)
- Multi-user isolation (later)
- Web frontend (later)

## Engineering Priorities

Before building new features, fix the foundation:

1. **Persist the HNSW index** — rebuilding on every startup won't scale. Save/load the index alongside the JSON stores.
2. **Add core tests** — hybrid search, category evolution, and the feedback loop need tests before being extended with new memory types.
3. **Fix or replace LLM integration** — make the provider configurable so the system works regardless of which API key is available.
4. **Refactor into package structure** — the current flat file layout needs to become a proper Python package before adding MCP server, new memory types, etc.

---

## What to Preserve from the Existing Codebase

There are ~4300 lines of working infrastructure. Don't rewrite it — refactor it into the new package structure and extend it.

- **`simple_memory_system.py`** → `src/memory/base.py` — the `MemorySystem` class is the foundation. Its storage, retrieval, and category management patterns are solid.
- **`enhanced_memory_system.py`** → `src/memory/enhanced.py` — dual embedding architecture stays. The `enhance_memory_system()` function that monkey-patches methods onto MemorySystem should probably become proper inheritance or composition.
- **`narrative_memory_system.py`** → `src/memory/narrative.py` — the `Narrative` and `NarrativePoint` classes are the foundation for intellectual arc tracking. Extend, don't replace.
- **`memory_feedback_loop.py`** → `src/memory/feedback.py` — the self-improvement logic stays. It should learn to evaluate the new memory types (positions, tensions, precedents) not just generic memories.
- **`memory_system_api.py`** → `src/api/` — keep as optional debugging/inspection interface.
- **Existing data files** — preserve the existing memory stores. They're useful for testing and development. The new memory types should coexist with the existing generic memories, not replace them.

---

## What NOT to Build

- No chat interface. The MCP server is the product. Clients already exist.
- No game mechanics, dice, domains, resources, or relationship ratings. Those belong to a different project.
- No system prompt management. The LLM client handles its own prompting. SynapticCore just provides memory tools.
- No user authentication for MVP. Single-user, local storage. Multi-user comes later.
- No web frontend. The existing FastAPI API can serve as a debugging/inspection tool, but the primary interface is MCP.

---

## Project Structure

```
synapticcore/
├── src/
│   ├── memory/
│   │   ├── base.py              # Core MemorySystem (from simple_memory_system.py)
│   │   ├── enhanced.py          # Dual embeddings (from enhanced_memory_system.py)
│   │   ├── narrative.py         # Narrative layer (from narrative_memory_system.py)
│   │   ├── feedback.py          # Self-improvement (from memory_feedback_loop.py)
│   │   ├── positions.py         # NEW: Position tracking
│   │   ├── tensions.py          # NEW: Tension detection/tracking
│   │   ├── precedents.py        # NEW: Precedent management
│   │   └── arcs.py              # NEW: Intellectual arc tracking
│   ├── retrieval/
│   │   ├── hybrid.py            # Existing hybrid search, refactored
│   │   └── activation.py        # NEW: Spreading activation
│   ├── mcp/
│   │   ├── server.py            # MCP server implementation
│   │   └── tools.py             # Tool definitions and handlers
│   └── storage/
│       ├── json_store.py        # File-based storage (existing pattern)
│       └── base.py              # Storage interface (for later PG swap)
├── tests/
├── config/
│   └── default.yaml
├── pyproject.toml
└── README.md
```

---

## What Success Looks Like

A working MCP server that:

1. **Connects** to Claude Desktop or any MCP-compatible client
2. **Stores** positions, tensions, and precedents as first-class memory types with dual embeddings
3. **Retrieves** relevant intellectual context using hybrid search with spreading activation — returning not just similar content but structurally adjacent material
4. **Tracks** how positions evolve across conversations — detecting reinforcement, shift, and reversal
5. **Surfaces** unexplored adjacent territory based on the user's intellectual topology
6. **Improves** its own organization over time through the existing feedback loop mechanisms

The test: after 10+ conversations through an MCP client, the system should be able to answer "how has my thinking about X changed?" and "what tensions keep coming back?" and "what haven't I explored that connects to what I'm working on now?" — not through keyword search but through genuine structural understanding of the user's intellectual landscape.

That's the thing nobody else has built. Build it.

---

## Key Architectural Principles

**Memory is topology, not database.** The system doesn't store records — it maintains a map of how ideas relate to each other in someone's thinking. The value is in the connections, not the entries.

**Contradiction is signal, not error.** When a tension recurs across conversations, that's important structural information. Don't resolve it. Track it. Surface it when relevant.

**Positions evolve; they don't update.** When someone changes their mind, the old position isn't deleted. The arc is preserved. The trajectory matters.

**Companionate attention, not surveillance.** The system knows the user deeply — but that knowledge is in service of better engagement, not profiling. The tools provide information that helps the LLM be a better thinking partner. The disposition is up to the protocol built on top.

**Yarn, not walls.** Keep the structural interventions minimal. The lightest possible demarcation that produces the desired effect. Every feature should be a constraint that enables something, not an apparatus that demands attention.
