# SynapticCore

**Memory that understands what you decided, what's still open, and why it matters.**

SynapticCore is an MCP server that gives AI agents structured memory of your decisions, tradeoffs, and constraints — not just facts. When your agent picks up where another left off, it inherits what was settled, what's still unresolved, and where the constraints came from. The result: agents that don't relitigate closed questions, know what's still open, and make better judgment calls on your behalf.

## The problem

Every AI memory system stores facts. "User likes Python." "Budget is $40K." "Prefers dark mode."

But that's not how people actually navigate decisions. You tried something and it didn't work. You committed to a direction after weighing tradeoffs. You settled a question, and that settlement constrains everything downstream. You keep circling back to the same unresolved tension.

Flat memory loses all of this. Your agent doesn't know *why* the budget is $40K, doesn't know the style question is still open, doesn't know you already ruled out the expensive option and why. So it asks again. Suggests things you've already rejected. Relitigates decisions you made weeks ago.

SynapticCore tracks three things:

- **Decisions** — what was settled, with confidence levels that evolve (tentative → held → committed)
- **Tradeoffs** — unresolved tensions between competing priorities, with engagement history
- **Constraints** — settled decisions that bind future choices, with records of whether they've held

These connect to form a **decision landscape** — a structured map of what's settled, what's open, and what constrains what. When an agent queries SynapticCore, it gets back not just relevant memories but *structurally adjacent* context, retrieved through spreading activation across this landscape.

## What it looks like

You're renovating your kitchen. Over three weeks, you talk to different agents about it.

**Week 1** — You discuss style with a design agent.

```python
store_interaction(
    decisions=[{
        "statement": "Leaning toward modern minimalist",
        "confidence": "tentative",
        "context": "Initial preference, haven't seen options yet"
    }],
    tradeoffs=[{
        "poles": ["Modern minimalist", "Farmhouse aesthetic"],
        "description": "Haven't landed on a style direction",
        "status": "active"
    }]
)
```

**Week 2** — A different agent helps with budget. It reads SynapticCore and *knows the style is still unsettled*.

```python
store_interaction(
    decisions=[{
        "statement": "Budget capped at $40K",
        "confidence": "committed",
        "context": "Contractor said moving plumbing adds $15K — decided to keep existing layout"
    }],
    constraints=[{
        "statement": "$40K cap — set by the plumbing decision",
        "held": True,
        "context": "Keeping existing layout redirected $15K into finishes budget"
    }]
)
```

**Week 3** — Your partner weighs in. The agent mediating that conversation inherits the full picture: budget is committed, style is still open, layout is a binding constraint.

**Week 4** — A shopping agent picks up. It queries SynapticCore with `depth="deep"` and gets back:

- The $40K constraint (direct match)
- The layout decision that established it (one hop)
- The now-resolved style tradeoff (connected through the layout decision)
- The partner's input that shifted the style (temporal chain with transition classification)

It doesn't suggest $50K countertops. It doesn't pull up pure minimalist options when the style shifted to transitional. It doesn't ask about the layout. It *knows*.

## How it works

SynapticCore uses a **dual embedding architecture** with **spreading activation** for retrieval:

```
Query → Hybrid Search (semantic + categorical + recency + associative)
           │
           ▼ (depth="deep")
       Spreading Activation
       ├── Seeds from search results across all types
       ├── Propagates through typed edges (decision↔tradeoff, decision↔constraint)
       ├── Lateral inhibition keeps results focused
       └── Returns nodes with activation paths explaining *how* it got there
```

Phase 4 **decision narrative tracking** classifies how your thinking evolves:

- **Reinforcement** — same direction, stronger ("even more sure about the layout")
- **Shift** — related but reframed ("still keeping layout, but now it's about the floors, not plumbing cost")
- **Reversal** — contradicts prior decision ("actually, tear out the wall")

Recurring tradeoffs that keep surfacing are flagged as **structural** — features of how you navigate decisions, not problems to solve.

## Installation

```bash
git clone https://github.com/gnarlynerd/synapticcore.git
cd synapticcore
python -m venv scagent_env
source scagent_env/bin/activate
pip install -r requirements.txt
```

### Run as MCP server

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "synapticcore": {
      "command": "python",
      "args": ["-m", "src.synapticcore.mcp.server"],
      "cwd": "/path/to/synapticcore"
    }
  }
}
```

Or run directly:

```bash
python -m src.synapticcore.mcp.server
```

## MCP Tools

| Tool | What it does |
|------|-------------|
| `store_interaction` | Record decisions, tradeoffs, and constraints from a conversation |
| `retrieve_relevant` | Get context from the decision landscape — surface, structural, or deep (with spreading activation) |
| `get_decision_narrative` | Trace how your approach to a topic evolved, with transition classifications |
| `find_adjacent` | Surface unexplored territory bordering the current conversation |
| `check_constraints` | Find constraints that apply to the current situation |
| `assess_depth` | Check whether there's deeper structure available for the current topic |

## Architecture

```
src/synapticcore/
  memory/       — base, enhanced, feedback, types, type_managers, arcs
  retrieval/    — spreading activation network
  mcp/          — FastMCP server + tool definitions
  storage/      — JSON persistence
  llm/          — configurable provider (DeepSeek, Anthropic, OpenAI)
tests/          — core, types, activation, arcs, MCP integration
```

### Memory types

- **Decisions** — statements with confidence levels (tentative/held/committed), context, evolution history, and links to related tradeoffs and constraints
- **Tradeoffs** — opposing poles with engagement history. Recurring tradeoffs are structural features, not bugs.
- **Constraints** — settled commitments with test history tracking whether they held or were violated

### Retrieval depths

- **`surface`** — semantic similarity only. Fast, conventional.
- **`structural`** — adds typed search across decisions, tradeoffs, and constraints.
- **`deep`** — spreading activation through the decision landscape. Returns results with activation paths showing how the system connected your query to each result. This is where SynapticCore is different from everything else.

## What makes this different

Every memory system in the current landscape — Mem0, Zep, Letta, LangMem, ChatGPT's memory, Claude's memory — stores facts and retrieves by similarity. SynapticCore tracks **how you navigate decisions over time** and retrieves through **structural adjacency in your decision landscape**.

The difference matters when:

- Multiple agents work on the same problem across sessions (handoff with cognitive state, not just task state)
- A companion agent needs to understand patterns in how you make decisions, not just what you decided
- Context from weeks ago is relevant not because it's semantically similar but because it *constrains* the current situation

No existing system combines spreading activation, dual embeddings, decision narrative tracking, and typed memory objects. See `docs/competitive_landscape.md` for the full analysis.

## Tech stack

- Python 3.8+
- Sentence Transformers (`all-MiniLM-L6-v2`)
- HNSWLib for vector indexing
- MCP Python SDK (FastMCP)
- JSON file persistence (single-user, local by design)

## Status

All core phases complete and running as an MCP server. Currently in use as the memory substrate for the [Lexington Stack](docs/lexington_stack_synthesis.md) project.

## Contributing

Contributions welcome. Areas of particular interest:

- Alternative storage backends (SQLite, PostgreSQL/pgvector)
- Multi-user isolation
- Additional MCP tool patterns
- Integration examples with agent frameworks (OpenClaw, LangChain, CrewAI)
- Benchmarking against other memory systems

## License

Apache License 2.0 — see LICENSE for details.

## Citation

```bibtex
@software{synapticcore2025,
  author = {Gerard Lynn},
  title = {SynapticCore: Cognitive Memory Architecture for AI Agents},
  year = {2025},
  url = {https://github.com/gnarlynerd/synapticcore}
}
```
