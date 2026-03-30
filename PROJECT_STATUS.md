# SynapticCore — Project Status

## What Is This

A cognitive memory architecture exposed as an MCP server. It tracks how a user's thinking evolves — decisions held and revised, recurring tradeoffs, binding constraints — not atomic facts. The core differentiator: memory as topology, not database.

Competitive analysis (`docs/`) found no existing system combining spreading activation + dual embeddings + self-improvement + intellectual arc tracking. Estimated strategic window: 12–18 months before convergence.

## Current State: Phase 5 Complete (2026-03-30)

All 6 MCP tools are implemented and the system is live.

| Phase | Commit | What landed |
|-------|--------|-------------|
| **0** | `97d9574` | HNSW persistence, LLM provider abstraction, core tests |
| **1** | `e3ee917` | Refactor into `src/synapticcore/` package |
| **2** | `adce977` | MCP server with 6 cognitive memory tools |
| **3** | `869f31d` | First-class memory types — decisions, tradeoffs, constraints |
| **4** | `cf1e1cd` | Intellectual arc tracking |
| **5** | `cb04ee2` | Spreading activation for topology-aware retrieval |
| — | `eafeb63` | Spec amendment: reframe as user narrative memory |
| — | `60a5525` | Fix numpy float32 serialization in retrieve_relevant |

## MCP Tools

| Tool | Purpose |
|------|---------|
| `store_interaction` | Record decisions, tradeoffs, constraints from conversations |
| `retrieve_relevant` | Hybrid search with optional spreading activation (surface/structural/deep) |
| `get_decision_narrative` | Trace how the user's approach to a topic evolved over time |
| `find_adjacent` | Surface unexplored territory bordering the current conversation |
| `check_constraints` | Find constraints that should apply to the current situation |
| `assess_depth` | Evaluate whether deeper intellectual structure is available |

## Architecture

```
src/synapticcore/
  memory/       — base, enhanced, feedback, types, type_managers, arcs
  retrieval/    — spreading activation network
  mcp/          — FastMCP server + tool definitions
  storage/      — JSON store, storage interface
  llm/          — configurable provider (DeepSeek, Anthropic, OpenAI)
tests/          — core, types, activation, arcs, MCP tools
```

### Retrieval pipeline

```
Query
  │
  ▼
Hybrid Search (4-phase)
  ├── Phase 1: Semantic similarity (HNSW KNN)
  ├── Phase 2: Category matching
  ├── Phase 3: Recency boost (exponential decay)
  └── Phase 4: Associative expansion
          │
          ▼ (depth="deep")
  Spreading Activation
  ├── Seed from semantic search across all types
  ├── Propagate through adjacency graph (typed edges)
  ├── Lateral inhibition for diversity
  └── Return activated nodes with paths
```

### Memory types

- **Decisions** (née positions) — statements with confidence, context, evolution history
- **Tradeoffs** (née tensions) — opposing poles with engagement history
- **Constraints** (née precedents) — commitments with test history tracking whether they held

Old internal prefixes (`pos:`, `ten:`, `pre:`) still used in activation graph IDs; mapped to user-facing vocabulary in MCP output.

## Tech Stack

- **Python 3.8+** with venv at `scagent_env/`
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for embeddings
- **HNSWLib** for vector indexing
- **MCP Python SDK** (`mcp`) — FastMCP server
- **FastAPI + Uvicorn** — legacy REST API, kept as debug tool
- **JSON files** for persistence
- **pytest** for tests

## Known Issues / Cleanup

- Internal ID prefixes (`pos:`, `ten:`, `pre:`) don't match external vocabulary (`decision`, `tradeoff`, `constraint`)
- LLM provider wired up but needs valid API key to function (DeepSeek 401s without one)
- No multi-user isolation (single-user, local — by design for MVP)
- JSON storage caps out around 10K memories practically

## Legacy Files

| File | Role | Status |
|------|------|--------|
| `simple_memory_system.py` | Original base memory system | Superseded by `src/synapticcore/memory/base.py` |
| `enhanced_memory_system.py` | Original dual-embedding layer | Superseded by `src/synapticcore/memory/enhanced.py` |
| `narrative_memory_system.py` | Original narrative tracking | Superseded by `src/synapticcore/memory/arcs.py` |
| `memory_feedback_loop.py` | Original feedback loop | Superseded by `src/synapticcore/memory/feedback.py` |
| `memory_system_api.py` | FastAPI REST API | Kept as debug tool |
| `chat_with_memory.py` | CLI chat interface | Kept but not the product |
| `memory.html` | Minimal web UI | Kept but not the product |
