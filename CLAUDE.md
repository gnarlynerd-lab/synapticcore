# SynapticCore

## What This Is

A cognitive memory architecture exposed as an MCP server. It tracks how a user's thinking evolves — positions held and broken, recurring tensions, intellectual arcs — not atomic facts. The core differentiator: memory as topology, not database.

See `docs/synapticcore-mcp-handoff.md` for the full spec and `PROJECT_STATUS.md` for current state.

## Architecture

~4300 lines of working Python infrastructure:
- `simple_memory_system.py` — base memory: embeddings (all-MiniLM-L6-v2), HNSW indexing, hybrid search, category management
- `enhanced_memory_system.py` — dual embedding spaces (content + category), outlier detection, clustering
- `narrative_memory_system.py` — narrative points, temporal ordering, intellectual arc foundation
- `memory_feedback_loop.py` — self-improvement: search tracking, quality metrics, category evolution
- `chat_with_memory.py` — CLI chat interface
- `memory_system_api.py` — FastAPI REST API
- `memory.html` — minimal web UI

Data lives in JSON files (`memory_store.json`, `enhanced_memory.json`, etc.). HNSW index rebuilds on startup.

## Target Architecture (from MCP handoff)

Refactor into a Python package and build an MCP server with 6 tools:
- `store_interaction` — store positions, tensions, precedents (not transcripts)
- `retrieve_relevant` — hybrid search with spreading activation, returns structurally adjacent material
- `get_intellectual_arc` — trace how positions evolved over time
- `find_adjacent` — surface unexplored territory bordering the user's topology
- `check_precedent` — look up established commitments and whether they held
- `assess_depth` — evaluate if conversation engages deeper intellectual structure

Target package structure:
```
src/
  memory/     — base, enhanced, narrative, feedback, positions, tensions, precedents, arcs
  retrieval/  — hybrid search, spreading activation
  mcp/        — server, tool definitions
  storage/    — JSON store, storage interface
tests/
```

## Tech Stack

- Python 3.8+, venv at `scagent_env/`
- sentence-transformers, hnswlib, numpy
- FastAPI + uvicorn (existing API, keep as debug tool)
- MCP Python SDK (`mcp`) — to be added
- pytest — to be added
- LLM provider: DeepSeek wired up but broken (401). Make provider configurable.

## Key Principles

- **Don't rewrite** — refactor existing ~4300 lines into the new package structure, then extend
- **Positions evolve, they don't update** — old positions are never deleted, arcs are preserved
- **Contradiction is signal, not error** — recurring tensions are structural features to track
- **Memory is topology** — value is in connections between ideas, not individual entries
- **Architecture doesn't dictate disposition** — MCP tools provide information, the LLM client decides how to use it

## What NOT to Build

- No chat interface (MCP server is the product, clients already exist)
- No game mechanics, dice, domains, resources
- No system prompt management
- No auth for MVP (single-user, local)
- No web frontend (FastAPI stays as debug tool)

## Commands

```bash
# Activate venv
source scagent_env/bin/activate

# Run existing CLI chat
python chat_with_memory.py

# Run existing API
python memory_system_api.py
```
