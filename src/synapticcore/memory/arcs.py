"""
Intellectual arc tracking.

Detects how positions evolve over time — reinforcement, shift, reversal —
and classifies trajectories as converging, oscillating, or deepening.
This is the novel capability that no other system implements.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

from .types import Position, Tension
from .type_managers import PositionManager, TensionManager

logger = logging.getLogger(__name__)

# Confidence levels ordered by strength
CONFIDENCE_ORDER = {"tentative": 0, "held": 1, "committed": 2}


class ArcTracker:
    """
    Detects and tracks how intellectual positions evolve over time.

    When a new position is stored, detect_relationship classifies it
    relative to existing positions: reinforcement (same stance, stronger),
    shift (related but different framing), or reversal (contradicts).

    get_arc returns the temporal chain of positions on a topic with
    transition classifications between each step.
    """

    def __init__(self, positions: PositionManager, tensions: TensionManager):
        self.positions = positions
        self.tensions = tensions

    def _cosine_similarity(self, emb1, emb2) -> float:
        """Compute cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        a = np.array(emb1) if isinstance(emb1, list) else emb1
        b = np.array(emb2) if isinstance(emb2, list) else emb2
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def detect_relationship(
        self, new_position: Position, relationship_hint: str = ""
    ) -> Optional[Dict]:
        """
        When a new position is stored, detect how it relates to existing ones.

        Args:
            new_position: The newly created position.
            relationship_hint: Optional hint from the calling LLM, e.g.
                "contradicts previous position on X" or "reinforces".
                Takes priority over automated detection.

        Returns:
            Dict with relationship info, or None if no related position found.
            {
                "type": "reinforcement" | "shift" | "reversal",
                "related_position_id": str,
                "related_position_statement": str,
                "confidence": float,  # how confident we are in the classification
                "explanation": str,
            }
        """
        if not self.positions.items:
            return None

        # Find the most similar existing position
        best_match = None
        best_similarity = 0.0

        for existing in self.positions.items:
            if existing.id == new_position.id:
                continue
            sim = self._cosine_similarity(new_position.embedding, existing.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_match = existing

        # Need a minimum similarity to consider it related
        # Lower threshold if the LLM provided a hint (it has conversational context we don't)
        min_similarity = 0.25 if relationship_hint else 0.4
        if best_match is None or best_similarity < min_similarity:
            return None

        # Classify the relationship
        # Priority 1: LLM-provided hint
        if relationship_hint:
            hint_lower = relationship_hint.lower()
            if any(w in hint_lower for w in ["contradict", "reverse", "opposite", "disagree", "changed mind"]):
                rel_type = "reversal"
            elif any(w in hint_lower for w in ["reinforce", "strengthen", "confirm", "agree", "same"]):
                rel_type = "reinforcement"
            elif any(w in hint_lower for w in ["shift", "evolve", "nuance", "modify", "refine"]):
                rel_type = "shift"
            else:
                # Hint doesn't map cleanly — fall through to automated detection
                rel_type = self._classify_automated(new_position, best_match, best_similarity)
        else:
            # Priority 2: Automated detection
            rel_type = self._classify_automated(new_position, best_match, best_similarity)

        # Record the relationship on the new position's evolution
        new_position.evolution.append({
            "timestamp": new_position.timestamp,
            "type": rel_type,
            "related_position_id": best_match.id,
            "similarity": round(best_similarity, 3),
            "previous_confidence": best_match.confidence,
            "new_confidence": new_position.confidence,
        })

        return {
            "type": rel_type,
            "related_position_id": best_match.id,
            "related_position_statement": best_match.statement,
            "similarity": round(best_similarity, 3),
            "confidence": self._relationship_confidence(rel_type, best_similarity, relationship_hint),
            "explanation": self._explain_relationship(rel_type, new_position, best_match, best_similarity),
        }

    def _classify_automated(
        self, new_pos: Position, existing: Position, similarity: float
    ) -> str:
        """
        Classify relationship without LLM hint.

        Heuristics:
        - High similarity (>0.8) + same/higher confidence = reinforcement
        - Medium similarity (0.5-0.8) = shift (related but reframed)
        - High similarity + confidence dropped to tentative from committed = potential reversal
        - Very high similarity (>0.9) + confidence went down = reversal signal

        Note: Pure embedding similarity can't distinguish "I agree with X" from
        "I disagree with X". This is a known limitation. LLM hints are the
        primary mechanism for reversal detection.
        """
        new_conf = CONFIDENCE_ORDER.get(new_pos.confidence, 0)
        old_conf = CONFIDENCE_ORDER.get(existing.confidence, 0)

        if similarity > 0.8:
            if new_conf >= old_conf:
                return "reinforcement"
            elif old_conf == 2 and new_conf == 0:
                # Committed → tentative on very similar topic is suspicious
                return "reversal"
            else:
                # Confidence went down but not dramatically
                return "shift"
        elif similarity > 0.5:
            return "shift"
        else:
            # Below 0.5 similarity — weakly related, treat as shift
            return "shift"

    def _relationship_confidence(
        self, rel_type: str, similarity: float, hint: str
    ) -> float:
        """How confident we are in the classification."""
        base = similarity * 0.5  # Similarity contributes half
        if hint:
            base += 0.4  # LLM hint adds significant confidence
        if rel_type == "reinforcement":
            base += 0.1  # Reinforcement is easiest to detect
        elif rel_type == "reversal" and not hint:
            base -= 0.2  # Reversal without hint is low confidence
        return round(min(max(base, 0.1), 1.0), 2)

    def _explain_relationship(
        self, rel_type: str, new_pos: Position, existing: Position, similarity: float
    ) -> str:
        """Generate a human-readable explanation."""
        if rel_type == "reinforcement":
            conf_change = ""
            if new_pos.confidence != existing.confidence:
                conf_change = f" (confidence: {existing.confidence} → {new_pos.confidence})"
            return (
                f"Reinforces a previous position{conf_change}. "
                f"Similarity: {similarity:.0%}."
            )
        elif rel_type == "shift":
            return (
                f"Shifts from a related position. "
                f"Previous: \"{existing.statement[:80]}\" "
                f"Similarity: {similarity:.0%}."
            )
        elif rel_type == "reversal":
            return (
                f"Appears to reverse a previous position. "
                f"Previous: \"{existing.statement[:80]}\" ({existing.confidence}) "
                f"Similarity: {similarity:.0%}."
            )
        return ""

    def get_arc(self, topic: str, time_range: Optional[str] = None) -> Dict:
        """
        Return the temporal chain of positions on a topic with transition
        classifications between each step.

        Args:
            topic: Subject to trace.
            time_range: Not yet implemented.

        Returns:
            {
                "topic": str,
                "positions": [...],  # chronological with transitions
                "trajectory": str,   # converging | oscillating | deepening | insufficient_data
                "recurring_tensions": [...],
                "summary": str,
            }
        """
        # Find relevant positions
        results = self.positions.search(topic, top_k=20)
        if not results:
            return {
                "topic": topic,
                "positions": [],
                "trajectory": "insufficient_data",
                "recurring_tensions": [],
                "summary": "No positions found on this topic.",
            }

        # Sort chronologically
        positions = sorted(results, key=lambda r: r["item"].timestamp)

        # Build the chain with transitions
        chain = []
        for i, r in enumerate(positions):
            pos = r["item"]
            entry = {
                "id": pos.id,
                "statement": pos.statement,
                "confidence": pos.confidence,
                "context": pos.context,
                "timestamp": pos.timestamp,
                "similarity_to_topic": round(r["similarity"], 3),
            }

            # Classify transition from previous position
            if i > 0:
                prev = positions[i - 1]["item"]
                sim = self._cosine_similarity(pos.embedding, prev.embedding)
                transition = self._classify_automated(pos, prev, sim)
                entry["transition_from_previous"] = {
                    "type": transition,
                    "similarity": round(sim, 3),
                    "confidence_change": f"{prev.confidence} → {pos.confidence}",
                }

            # Include any evolution history recorded at storage time
            if pos.evolution:
                entry["evolution_history"] = pos.evolution

            chain.append(entry)

        # Classify the overall trajectory
        trajectory = self.classify_trajectory([r["item"] for r in positions])

        # Find recurring tensions related to this topic
        ten_results = self.tensions.search(topic, top_k=5)
        recurring = []
        for r in ten_results:
            t = r["item"]
            if len(t.engagement_history) >= 2 or t.status == "active":
                recurring.append({
                    "id": t.id,
                    "poles": t.poles,
                    "description": t.description,
                    "status": t.status,
                    "engagements": len(t.engagement_history),
                    "similarity": round(r["similarity"], 3),
                })

        # Generate summary
        summary = self._summarize_arc(chain, trajectory, recurring)

        return {
            "topic": topic,
            "positions": chain,
            "trajectory": trajectory,
            "recurring_tensions": recurring,
            "summary": summary,
        }

    def classify_trajectory(self, positions: List[Position]) -> str:
        """
        Classify a sequence of positions as converging, oscillating,
        deepening, or diverging.

        - Converging: confidence increases over time, positions stabilize
        - Oscillating: confidence or stance flips back and forth
        - Deepening: positions become more nuanced (shifts with increasing confidence)
        - Diverging: positions spread to new territory
        - Insufficient data: fewer than 2 positions
        """
        if len(positions) < 2:
            return "insufficient_data"

        confidences = [CONFIDENCE_ORDER.get(p.confidence, 0) for p in positions]

        # Check for oscillation: confidence goes up then down (or vice versa) multiple times
        direction_changes = 0
        for i in range(1, len(confidences)):
            if i >= 2:
                prev_dir = confidences[i - 1] - confidences[i - 2]
                curr_dir = confidences[i] - confidences[i - 1]
                if (prev_dir > 0 and curr_dir < 0) or (prev_dir < 0 and curr_dir > 0):
                    direction_changes += 1

        if direction_changes >= 2:
            return "oscillating"

        # Check for convergence: confidence generally increases
        if len(positions) >= 2:
            # Compare first half average confidence to second half
            mid = len(confidences) // 2
            first_half_avg = sum(confidences[:mid]) / max(mid, 1)
            second_half_avg = sum(confidences[mid:]) / max(len(confidences) - mid, 1)

            if second_half_avg > first_half_avg + 0.3:
                return "converging"

        # Check for deepening: positions shift but maintain or increase confidence
        if len(positions) >= 3:
            similarities = []
            for i in range(1, len(positions)):
                sim = self._cosine_similarity(positions[i].embedding, positions[i - 1].embedding)
                similarities.append(sim)

            avg_similarity = sum(similarities) / len(similarities)
            # Moderate similarity (shifting) + stable/increasing confidence = deepening
            if 0.4 < avg_similarity < 0.85 and second_half_avg >= first_half_avg:
                return "deepening"

        # Default
        if len(positions) >= 2 and confidences[-1] >= confidences[0]:
            return "converging"
        return "deepening"  # positions exist but don't fit other patterns clearly

    def detect_recurring_tensions(self, min_engagements: int = 2) -> List[Dict]:
        """Find tensions that keep surfacing across conversations."""
        recurring = self.tensions.get_recurring(min_engagements=min_engagements)
        return [
            {
                "id": t.id,
                "poles": t.poles,
                "description": t.description,
                "status": t.status,
                "engagement_count": len(t.engagement_history),
                "last_engaged": t.engagement_history[-1]["timestamp"] if t.engagement_history else t.timestamp,
            }
            for t in recurring
        ]

    def _summarize_arc(self, chain: List[Dict], trajectory: str, recurring: List[Dict]) -> str:
        """Generate a brief narrative summary of the arc."""
        if not chain:
            return "No positions found."

        n = len(chain)
        first = chain[0]
        last = chain[-1]

        parts = [f"{n} position{'s' if n > 1 else ''} found."]

        if n >= 2:
            parts.append(f"Trajectory: {trajectory}.")
            parts.append(
                f"From \"{first['statement'][:60]}\" ({first['confidence']}) "
                f"to \"{last['statement'][:60]}\" ({last['confidence']})."
            )

            # Count transitions
            transitions = [c.get("transition_from_previous", {}).get("type") for c in chain if "transition_from_previous" in c]
            if transitions:
                from collections import Counter
                counts = Counter(transitions)
                parts.append(f"Transitions: {dict(counts)}.")

        if recurring:
            parts.append(f"{len(recurring)} recurring tension{'s' if len(recurring) > 1 else ''} nearby.")

        return " ".join(parts)
