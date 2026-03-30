# SynapticCore — Spec Amendment: From Intellectual Arcs to User Narrative Memory

## Context

This amends the Phase 3–5 roadmap for SynapticCore. The architecture is unchanged. The framing shifts.

The original spec describes SynapticCore as a system for tracking **intellectual evolution** — how a user's positions on topics develop over time, with tensions as unresolved philosophical oppositions and precedents as established intellectual commitments. That framing reflects the builder's use case (an intellectual working through complex ideas across sessions) but limits the system's appeal and market positioning.

This amendment reframes the same abstractions for **user narrative continuity** — how anyone navigates decisions, tradeoffs, and commitments over time when working with an AI agent. The internal architecture (typed dataclasses, transition classification, spreading activation) stays exactly the same. The vocabulary, examples, documentation, and developer-facing API descriptions shift.

---

## The Core Reframe

### What stays the same

- Positions, tensions, and precedents as first-class typed objects
- Phase 3: typed dataclasses, type-specific managers, linked structures
- Phase 4: transition classification (reinforcement, shift, reversal), temporal chains, trajectory analysis
- Phase 5: spreading activation with decay, lateral inhibition, multi-hop traversal
- The HNSW index with type-prefixed IDs
- The dual embedding architecture
- The MCP server interface

### What changes

The **language** and **framing** around these abstractions.

| Old framing | New framing |
|---|---|
| Intellectual position | Decision or commitment — something the user has settled on, with a confidence level that can evolve |
| Intellectual tension | Unresolved tradeoff — two things pulling against each other that the user hasn't resolved yet |
| Intellectual precedent | Settled constraint — a decision that was made and should bind future actions unless explicitly revisited |
| Intellectual arc | Decision narrative — how the user's approach to a problem evolved over time |
| Topology of intellectual commitments | Decision landscape — the structured map of what's settled, what's open, and what constrains what |

The user never sees any of these terms. They experience an agent that **doesn't relitigate settled decisions**, **knows what's still open**, and **understands where constraints came from**.

---

## Revised Phase 3: Typed User Narrative Objects

Each type gets its own dataclass, manager, and section in the JSON store.

### Decision (formerly Position)

```python
@dataclass
class Decision:
    id: str                          # "decision:72"
    statement: str                   # "We're keeping the existing kitchen layout"
    confidence: ConfidenceLevel      # tentative | held | committed
    context: str                     # "Contractor said moving plumbing would cost $15K more"
    created_at: datetime
    updated_at: datetime
    related_tradeoffs: list[str]     # IDs of related Tradeoff objects
    related_constraints: list[str]   # IDs of related Constraint objects
    supersedes: Optional[str]        # ID of decision this replaced
    confidence_history: list[tuple[datetime, ConfidenceLevel]]  # evolution over time
```

The `DecisionManager` supports:
- `update_confidence(id, new_level)` — e.g., moving from tentative to committed
- `supersede(old_id, new_decision)` — creates a new decision that explicitly replaces an old one, preserving the chain
- `find_related(id)` — returns linked tradeoffs and constraints
- `get_active()` — returns all decisions still in play (not superseded)

### Tradeoff (formerly Tension)

```python
@dataclass
class Tradeoff:
    id: str                          # "tradeoff:15"
    poles: tuple[str, str]           # ("Modern minimalist style", "Farmhouse aesthetic")
    description: str                 # "Haven't agreed on kitchen style — user shifted from minimalist to farmhouse but partner prefers minimalist"
    status: TradeoffStatus           # active | explored | resolved
    created_at: datetime
    updated_at: datetime
    engagement_count: int            # how many times this has come up
    related_decisions: list[str]     # decisions bearing on this tradeoff
    resolution: Optional[str]        # if resolved, what was decided
```

The `TradeoffManager` supports:
- `engage(id)` — increments engagement count, updates timestamp
- `resolve(id, resolution_statement)` — marks resolved and records outcome
- `find_recurring(threshold)` — finds tradeoffs that keep surfacing (engagement_count above threshold)
- `get_active()` — returns unresolved tradeoffs

Recurring tradeoffs are **features, not bugs**. A tradeoff the user keeps returning to (budget vs. quality, speed vs. thoroughness, independence vs. collaboration) is a structural feature of how they navigate decisions. The system flags these rather than trying to resolve them.

### Constraint (formerly Precedent)

```python
@dataclass
class Constraint:
    id: str                          # "constraint:8"
    statement: str                   # "Budget is capped at $40K after the plumbing decision"
    source_decision: Optional[str]   # which decision established this constraint
    held: bool                       # has this constraint been respected or violated
    created_at: datetime
    tested_at: list[datetime]        # each time the constraint was relevant
    test_outcomes: list[bool]        # did it hold each time
    context: str                     # "The $40K cap came from deciding to keep existing layout and redirect savings"
```

The `ConstraintManager` supports:
- `test(id, held: bool)` — records that the constraint was relevant and whether it held
- `find_binding(context)` — given a current situation, find constraints that should apply
- `violated(id)` — records that the constraint was broken, flags for user awareness

### Storage

All three types live in the same JSON store with separate sections:

```json
{
  "decisions": [...],
  "tradeoffs": [...],
  "constraints": [...],
  "metadata": {
    "version": "0.3.0",
    "last_updated": "2026-03-30T..."
  }
}
```

Embeddings go in the shared HNSW index with type-prefixed IDs (`decision:72`, `tradeoff:15`, `constraint:8`).

---

## Revised Phase 4: Decision Narrative Tracking

When a new decision comes in that's semantically close to an existing one, classify the relationship:

- **Reinforcement** — same direction, stronger commitment. "I'm even more sure we should keep the layout."
- **Shift** — related but reframed. "I still want to keep the layout, but now it's about preserving the wood floors, not the plumbing cost."
- **Reversal** — contradicts the prior decision. "Actually, let's tear out the wall. The open concept is worth the $15K."

Build temporal chains: Decision A (March) → Decision B (April) → Decision C (June). Classify trajectories:

- **Converging** — decisions narrowing toward a stable position
- **Oscillating** — going back and forth (common with active tradeoffs)
- **Deepening** — same direction but with richer reasoning

### Recurring tradeoff detection

When a tradeoff's `engagement_count` crosses a threshold AND it remains unresolved, classify it as a **structural tradeoff** — a recurring feature of how this user navigates decisions, not a problem to be solved. Examples: budget vs. quality, career growth vs. work-life balance, moving fast vs. getting it right.

### MCP tool upgrades

`get_intellectual_arc` → `get_decision_narrative`
- Input: topic string
- Output: chronological chain of related decisions with transition classifications and trajectory analysis
- v1 (current): "here are related items sorted by date"
- v2 (Phase 4): "here's how your approach to X evolved, with transitions classified and trajectory labeled"

`store_interaction` gains a `relationship_hint` field:
- The calling LLM can annotate: "this contradicts what they said before about the budget" or "this reinforces the layout decision"
- Supplements automated semantic detection, which can't reliably distinguish reinforcement from reversal (both are semantically close)

---

## Revised Phase 5: Spreading Activation for User Narrative

Replace the naive category-based associative expansion with real spreading activation.

### The activation graph

Nodes: decisions, tradeoffs, constraints, plus category nodes from the existing system.

Edges built from:
- Decision ↔ Tradeoff links (a decision bears on a tradeoff)
- Decision ↔ Constraint links (a decision established or is constrained by)
- Tradeoff ↔ Constraint links (a constraint limits how a tradeoff can resolve)
- Semantic similarity edges (above threshold)
- Category co-membership

### Activation dynamics

- Activation propagates through edges, decaying per hop
- Old connections fade with time (temporal decay on edge weights)
- Strongly activated nodes suppress weakly related alternatives (lateral inhibition)
- Result: focused retrieval, not retrieval soup

### Practical result

When an agent asks about "kitchen budget," the system returns:
- The $40K cap constraint (direct match)
- The layout decision that established the cap (one hop through constraint → source_decision)
- The still-active style tradeoff (connected through the layout decision)
- The partner disagreement (connected through the style tradeoff)
- NOT: the unrelated vacation budget conversation from two months ago (suppressed by lateral inhibition)

`retrieve_relevant` in deep mode returns results with **activation paths** — showing how the system got from the query to each result. This is the "stairs evokes slip" pattern from Schank's PAM.

`find_adjacent` uses the same network to identify weakly-activated-but-reachable territory — gaps in the user's decision landscape. "You've decided on layout and budget, but the countertop material question connects to both and hasn't been addressed."

---

## Agent Handoff Architecture

This is the near-term product value, not full mutual modeling between agents.

### The handoff problem

Agent A works on a task, builds up structured understanding. Agent B picks up the task. Currently, Agent B gets either a transcript dump (too much noise) or a summary (too little structure). Neither preserves what matters: what's been decided, what's still open, what constrains future choices.

### SynapticCore handoff

Agent A writes to SynapticCore during its work:
- Decisions it reached or observed the user reach
- Tradeoffs it identified as unresolved
- Constraints it discovered or established

Agent B reads from SynapticCore when it picks up:
- Active decisions (what's settled)
- Active tradeoffs (what's still open)
- Binding constraints (what limits the options)
- The user's decision narrative on this topic (how they got here)

Agent B inherits **cognitive state**, not just task state. It knows not just "the user wants X" but "the user shifted from Y to X because of Z, and there's still an unresolved tension with W."

### The human model

Both agents share the same SynapticCore model of the human they're working for. Agent B doesn't start cold — it inherits Agent A's accumulated understanding of:
- How this user navigates decisions (do they commit fast or deliberate?)
- What structural tradeoffs keep recurring for them
- What constraints they've set for themselves
- Where their confidence levels sit on open questions

This is the bridge between the companion use case (where the model gets built through conversation) and the agent use case (where the model enables better autonomous judgment).

---

## MCP Tool Interface — Revised Names and Descriptions

The MCP tools keep the same internal functionality but get user-narrative-oriented names and descriptions.

| Current tool | Revised tool | Description |
|---|---|---|
| `store_interaction` | `store_interaction` | Store decisions, tradeoffs, and constraints from a conversation. Each decision tracks confidence evolution. Each tradeoff tracks engagement history. Each constraint tracks whether it held under pressure. |
| `retrieve_relevant` | `retrieve_relevant` | Retrieve relevant user narrative context. In deep mode, returns activation paths showing how results connect to the query through the decision landscape. |
| `get_intellectual_arc` | `get_decision_narrative` | Trace how the user's approach to a topic evolved over time. Returns chronological chain of decisions with transition classifications (reinforcement, shift, reversal) and trajectory analysis (converging, oscillating, deepening). |
| `check_precedent` | `check_constraints` | Find constraints that should apply to the current situation. Returns constraint objects with test histories showing whether they've held or been violated. |
| `find_adjacent` | `find_adjacent` | Identify unaddressed territory bordering the current conversation. Finds tradeoffs that haven't been explored, decisions that connect to the current context but haven't been made, gaps in the decision landscape. |
| `assess_depth` | `assess_depth` | Evaluate whether the conversation is engaging with the user's decision landscape. Information tool — tells the LLM "there's more relevant context here if you want to go deeper." |

---

## What This Does NOT Change

- The Lexington Game can still use the original vocabulary (positions, tensions, precedents) — it's a philosophical game, the intellectual framing fits
- The competitive landscape analysis and patent positioning remain valid — the architecture is unchanged
- The BAC layer design is unaffected
- The Reading Commons vision is unaffected
- Internal code can retain whatever naming makes the architecture clearest

This amendment changes **how SynapticCore is presented to developers and users**, not how it works. The kitchen renovation is the same architecture as the intellectual arc. The vocabulary is what makes one feel relevant and the other feel academic.

---

## User Narrative Examples for Documentation

### Kitchen renovation (multi-week, multi-agent)

Week 1: User discusses style with Agent A. Agent A stores:
- Decision: "Leaning toward modern minimalist" (tentative)
- Tradeoff: "Modern minimalist vs. farmhouse" (active)

Week 2: User talks to Agent B about budget. Agent B reads SynapticCore, knows the style is unsettled. Stores:
- Decision: "Budget capped at $40K" (committed)
- Constraint: "$40K cap — came from the plumbing assessment"

Week 3: Partner weighs in via Agent A. Agent A reads current state, knows budget is committed and style is still open. Stores:
- Decision: "Going with transitional style — compromise" (held)
- Resolution on the style tradeoff

Week 4: Agent C (shopping agent) picks up. Reads SynapticCore. Knows: transitional style (held, not committed — might shift), $40K budget (committed constraint, tested and held), layout is staying (committed constraint, established week 1). Shops accordingly. Doesn't suggest $50K countertops. Doesn't pull up pure minimalist options.

### Career transition (months-long, companion use case)

Month 1: User exploring whether to leave current job. Decision: "Considering leaving" (tentative). Tradeoff: "Financial security vs. creative fulfillment" (active).

Month 3: User got a raise. Decision shifts: "Staying for now but keeping options open" (held). The system classifies this as a **shift** from the month 1 decision, not a reversal — the underlying tradeoff is still active.

Month 5: User's side project gets traction. Decision: "Going independent" (committed). System classifies this as a **convergence** — the trajectory moved from tentative exploration to commitment over 5 months, with one oscillation when the raise came.

The agent that helps with month 6 planning knows this entire arc. It doesn't ask "have you thought about going independent?" — it knows that decision was made, knows the financial security tradeoff was the main drag, knows the raise was a data point that slowed but didn't stop the trajectory.

---

## Implementation Priority

1. **Phase 3 dataclasses and managers** — this is the foundation. Ship with the revised naming (Decision, Tradeoff, Constraint).
2. **MCP tool renames** — `get_decision_narrative`, `check_constraints`. Low effort, high signal.
3. **Phase 4 transition classification** — the thing nobody else has. Start with the `relationship_hint` field so calling LLMs can annotate, then build the automated detection.
4. **Phase 5 spreading activation** — the thing that makes retrieval intelligent rather than just semantic.
5. **Agent handoff protocol** — the near-term product value. Document how agents read/write SynapticCore state during handoffs.

---

## One More Thing

The "intellectual arc" framing isn't wrong. It's one use case of a general architecture. The user who tracks how their thinking about AI safety evolved over six months and the user who tracks how their kitchen renovation decisions unfolded over three weeks are using the same system. The first framing appeals to academics and researchers. The second framing appeals to everyone.

Ship the second. The first will find you.
