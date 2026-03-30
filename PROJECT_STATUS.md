# SynapticCore — Project Status

## What Is This

SynapticCore is a cognitive memory architecture for LLM agents. It combines dual-embedding spaces (semantic + categorical), associative retrieval with spreading activation, self-improving category evolution, and narrative tracking of intellectual development. The core idea: treat users as evolving intellectual agents, not preference profiles.

A thorough competitive analysis (in `docs/`) found no existing system that combines all of these — spreading activation + dual embeddings + self-improvement + intellectual arc tracking. Estimated strategic window: 12–18 months before convergence.

## Tech Stack

- **Python 3.8+** with venv at `scagent_env/`
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for embeddings
- **HNSWLib** for vector indexing / approximate nearest neighbor search
- **FastAPI + Uvicorn** for the REST API
- **DeepSeek API** for LLM generation (requires `DEEPSEEK_API_KEY`)
- **JSON files** for persistence (memory stores, feedback history, conversations)

## File Map

| File | Role |
|------|------|
| `simple_memory_system.py` | Base memory: embeddings, HNSW indexing, semantic/hybrid search, category management with versioning |
| `enhanced_memory_system.py` | Dual-embedding layer: category embeddings, similarity scoring, outlier detection, relationship discovery |
| `memory_feedback_loop.py` | Self-improvement: search tracking, quality metrics, category evolution suggestions, auto-refinement |
| `narrative_memory_system.py` | Narrative points, intellectual arc tracking, temporal organization, narrative search |
| `chat_with_memory.py` | Interactive CLI integrating all systems with DeepSeek LLM |
| `memory_system_api.py` | FastAPI REST API exposing all memory operations |
| `memory.html` | Basic web UI visualization |
| `docs/compass_artifact_wf-*.md` | Competitive analysis (30+ products, 15+ papers, patent landscape) |

## What's Working

- **Core memory system** — full CRUD, UUID-based, JSON persistence
- **Semantic search** — sentence-transformer embeddings + HNSW KNN queries
- **Hybrid search** — 4-phase retrieval: semantic similarity, category matching, recency boost (exponential decay), associative expansion (spreading activation)
- **Category management** — dynamic creation, version history, deprecation workflow, relationship mapping
- **Dual embeddings** — category-level aggregated embeddings, cross-category similarity, outlier detection, new category suggestions via clustering
- **Feedback loop** — search pattern tracking, failed search logging, category coherence metrics, improvement suggestions (merge/split/rename/create), auto-apply
- **Narrative system** — narrative creation from memories, confidence-scored points, temporal ordering, narrative search
- **API layer** — full FastAPI endpoints for all operations
- **CLI chat** — interactive mode with `!search`, `!feedback`, `!metrics` commands

## What's Partial or Broken

- **LLM integration** — DeepSeek API calls are wired up but stored data shows 401 auth errors. Requires a valid `DEEPSEEK_API_KEY` to function.
- **Narrative generation** — structure is in place but LLM-driven synthesis is incomplete
- **Web UI** — `memory.html` exists but is minimal

## What Doesn't Exist Yet

- **Agent framework** — README says Phase 1 "in progress" but no agent reasoning loop, planning, or autonomous behavior is implemented
- **Tests** — zero test files
- **Scalable storage** — everything is single-file JSON; no database backend
- **Vector index persistence** — HNSW index rebuilds from scratch on every startup
- **Multi-user isolation** — none

## Data

Several JSON stores exist with real data:
- `memory_store.json` (~623 KB, ~19K lines) — primary store
- `memory_store_v3.json` (~203 KB) — versioned store
- `enhanced_memory.json` (~72 KB) — dual-embedding data
- `conversation_memory.json` (~43 KB) — chat history
- `feedback_history.json` (~14 KB) — metrics/feedback

## Architecture at a Glance

```
User Input (Chat / API)
    │
    ▼
Memory System ──→ Embedding Model ──→ Vector Index (HNSW)
    │                                       │
    ├── Category Manager (versioned)        │
    │                                       │
    ▼                                       ▼
Hybrid Search ◄────────────────────────────┘
    ├── Phase 1: Semantic similarity (KNN)
    ├── Phase 2: Category matching
    ├── Phase 3: Recency boost (exp decay)
    └── Phase 4: Associative expansion (spreading activation)
            │
            ▼
    Feedback Loop ──→ Metrics ──→ Suggestions ──→ Auto-refinement
            │
            ▼
    Narrative Layer ──→ Intellectual arc tracking
```

## README Roadmap (as stated)

| Phase | Focus | Status |
|-------|-------|--------|
| **1** | Core agent framework, memory-agent interface, basic web UI | Stated "in progress" — memory is solid, agent loop not started |
| **2** | Autonomous knowledge organization, interactive exploration, self-improvement | Not started |
| **3** | Research assistant, enhanced visualization, evaluation framework | Not started |

## Biggest Gaps to Close Next

1. **Fix LLM integration** — get DeepSeek auth working (or swap to another provider) so chat, narrative generation, and LLM-driven suggestions actually function
2. **Add tests** — nothing is tested; the core search and category logic is complex enough to warrant it
3. **Start the agent loop** — the memory system is the foundation, but the agent reasoning/planning layer is the product
4. **Persist the HNSW index** — rebuilding on every startup won't scale
5. **Consider a real database** — JSON files cap out around 10K memories practically
