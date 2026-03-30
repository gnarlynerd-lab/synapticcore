"""
Memory feedback loop for self-improvement.

Refactored from memory_feedback_loop.py — takes MemorySystem and EnhancedMemory
as composed references instead of expecting monkey-patched methods.
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict


class MemoryFeedbackLoop:
    """
    Feedback mechanisms for memory system self-improvement.
    Tracks search patterns, evaluates category quality, and suggests improvements.
    """

    def __init__(self, memory_system, enhanced_memory=None,
                 feedback_log_path="feedback_history.json"):
        """
        Args:
            memory_system: MemorySystem instance
            enhanced_memory: Optional EnhancedMemory instance for category analysis.
                           If None, category distribution/relationship methods won't be available.
            feedback_log_path: Path to store feedback history
        """
        self.memory_system = memory_system
        self.enhanced = enhanced_memory
        self.feedback_log_path = feedback_log_path
        self.feedback_history = self._load_feedback_history()

        # Performance tracking
        self.search_patterns = defaultdict(int)
        self.category_hits = defaultdict(int)
        self.failed_searches = []

        # Quality metrics
        self.category_coherence = {}
        self.category_utility = {}

        self.update_quality_metrics()

    def _load_feedback_history(self):
        if os.path.exists(self.feedback_log_path):
            try:
                with open(self.feedback_log_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading feedback history: {e}")
        return {"feedback_events": [], "metrics_history": []}

    def _save_feedback_history(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.feedback_log_path)), exist_ok=True)
            with open(self.feedback_log_path, 'w') as f:
                json.dump(self.feedback_history, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback history: {e}")

    def chatbot_memory_search(self, query: str, categories: List[str] = None,
                              result_count: int = 3, use_enhanced_search: bool = False) -> Dict:
        """Search memory system and record patterns for feedback."""
        # Record search pattern
        search_terms = set(query.lower().split())
        for term in search_terms:
            if len(term) > 3:
                self.search_patterns[term] += 1

        # Perform search
        start_time = time.time()
        if use_enhanced_search:
            results = self.memory_system.enhanced_hybrid_search(query, categories=categories, top_k=result_count)
        else:
            results = self.memory_system.hybrid_search(query, categories=categories, top_k=result_count)
        search_time = time.time() - start_time

        # Track category hits
        found_categories = set()
        for result in results:
            memory = result.get("memory", result)
            for category in memory.get("categories", []):
                found_categories.add(category)
                self.category_hits[category] += 1

        # Track failed searches
        avg_score = 0
        if results:
            if "combined_score" in results[0]:
                avg_score = sum(r.get("combined_score", 0) for r in results) / len(results)
            elif "similarity" in results[0]:
                avg_score = sum(r.get("similarity", 0) for r in results) / len(results)
            if avg_score < 0.5:
                self.failed_searches.append({
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "avg_score": avg_score
                })

        # Format results
        formatted_results = []
        for result in results:
            memory = result.get("memory", result)
            score = result.get("combined_score", result.get("similarity", 0))
            retrieval_method = result.get("retrieval_method", "search")

            content = memory["content"]
            if memory.get("metadata", {}).get("type") == "conversation":
                pass
            elif "Assistant:" in content:
                parts = content.split("Assistant:", 1)
                content = parts[0].strip()
                if content.startswith("User:"):
                    content = content[5:].strip()

            if len(content.strip()) < 5:
                continue

            formatted_results.append({
                "content": content,
                "categories": memory.get("categories", []),
                "relevance": score,
                "timestamp": memory.get("timestamp", ""),
                "retrieval_method": retrieval_method
            })

        # Log search event
        self.feedback_history["feedback_events"].append({
            "type": "search",
            "query": query,
            "categories_filter": categories,
            "results_count": len(results),
            "found_categories": list(found_categories),
            "avg_score": avg_score,
            "search_time": search_time,
            "timestamp": datetime.now().isoformat(),
            "search_method": "enhanced" if use_enhanced_search else "standard"
        })
        self._save_feedback_history()

        return {
            "results": formatted_results,
            "found_categories": list(found_categories),
            "search_time": search_time,
            "query_terms": list(search_terms),
            "analysis": None
        }

    def log_category_feedback(self, category_name: str, is_useful: bool,
                              memory_id: Optional[int] = None) -> None:
        self.feedback_history["feedback_events"].append({
            "type": "category_feedback",
            "category": category_name,
            "is_useful": is_useful,
            "memory_id": memory_id,
            "timestamp": datetime.now().isoformat()
        })
        self._save_feedback_history()

    def log_memory_categorization(self, memory_id: int, suggested_categories: List[str],
                                  accepted_categories: List[str]) -> None:
        self.feedback_history["feedback_events"].append({
            "type": "categorization_feedback",
            "memory_id": memory_id,
            "suggested_categories": suggested_categories,
            "accepted_categories": accepted_categories,
            "acceptance_rate": len(set(suggested_categories) & set(accepted_categories)) /
                              max(1, len(suggested_categories)),
            "timestamp": datetime.now().isoformat()
        })
        self._save_feedback_history()

    def update_quality_metrics(self) -> Dict:
        """Update quality metrics for all categories."""
        if not self.memory_system.categories or not self.memory_system.memories:
            return {}

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "category_metrics": {},
            "system_metrics": {
                "total_categories": len(self.memory_system.categories),
                "total_memories": len(self.memory_system.memories),
                "categorized_memories_percent": 0,
                "avg_categories_per_memory": 0,
                "avg_coherence": 0
            }
        }

        all_coherence_values = []

        for category_name in self.memory_system.categories:
            distribution = None
            if self.enhanced:
                distribution = self.enhanced.get_category_distribution(category_name)
            if not distribution or distribution["member_count"] < 2:
                continue

            coherence = 0
            if distribution["variance"] and distribution["variance"] > 0:
                coherence = 1.0 / distribution["variance"]

            utility = self.category_hits.get(category_name, 0)

            self.category_coherence[category_name] = coherence
            self.category_utility[category_name] = utility

            metrics["category_metrics"][category_name] = {
                "member_count": distribution["member_count"],
                "coherence": coherence,
                "utility": utility,
                "variance": distribution.get("variance", 0)
            }

            if coherence > 0:
                all_coherence_values.append(coherence)

        categorized_count = 0
        category_count_sum = 0
        for memory in self.memory_system.memories:
            categories = memory.get("categories", [])
            if categories:
                categorized_count += 1
                category_count_sum += len(categories)

        if self.memory_system.memories:
            metrics["system_metrics"]["categorized_memories_percent"] = (
                categorized_count / len(self.memory_system.memories) * 100
            )
        if categorized_count > 0:
            metrics["system_metrics"]["avg_categories_per_memory"] = category_count_sum / categorized_count
        if all_coherence_values:
            metrics["system_metrics"]["avg_coherence"] = sum(all_coherence_values) / len(all_coherence_values)

        self.feedback_history["metrics_history"].append(metrics)
        self._save_feedback_history()
        return metrics

    def suggest_improvements(self) -> Dict:
        """Suggest improvements to the category system based on feedback."""
        suggestions = {
            "merge_suggestions": [],
            "split_suggestions": [],
            "new_category_suggestions": [],
            "rename_suggestions": [],
            "timestamp": datetime.now().isoformat()
        }

        # 1. Merge suggestions (need enhanced memory for category relationship discovery)
        if self.enhanced:
            relationships = self.enhanced.discover_category_relationships(min_similarity=0.75)
            for rel in relationships:
                cat1, cat2 = rel["source"], rel["target"]
                cat1_events = [e for e in self.feedback_history["feedback_events"]
                               if e["type"] == "search" and cat1 in e.get("found_categories", [])]
                cat2_events = [e for e in self.feedback_history["feedback_events"]
                               if e["type"] == "search" and cat2 in e.get("found_categories", [])]
                overlap_events = [e for e in cat1_events if e["query"] in [ce["query"] for ce in cat2_events]]

                if len(overlap_events) >= 3 or (len(cat1_events) > 0 and len(overlap_events) / len(cat1_events) > 0.5):
                    suggestions["merge_suggestions"].append({
                        "categories": [cat1, cat2],
                        "similarity": rel["similarity"],
                        "reason": f"High similarity ({rel['similarity']:.2f}) and frequently used together"
                    })

        # 2. Split suggestions
        if self.enhanced:
            for cat_name, coherence in self.category_coherence.items():
                if coherence > 0 and coherence < 0.2:
                    outliers = self.enhanced.find_category_outliers(cat_name, threshold=1.2)
                    if len(outliers) >= 3:
                        suggestions["split_suggestions"].append({
                            "category": cat_name,
                            "coherence": coherence,
                            "outlier_count": len(outliers),
                            "reason": f"Low coherence ({coherence:.2f}) with {len(outliers)} outlier memories"
                        })

        # 3. New category suggestions from search patterns
        common_terms = sorted(self.search_patterns.items(), key=lambda x: x[1], reverse=True)
        existing_categories = set(self.memory_system.categories.keys())

        for term, count in common_terms[:10]:
            if count >= 3 and term not in existing_categories and len(term) > 3:
                matching_memories = []
                for i, memory in enumerate(self.memory_system.memories):
                    if term.lower() in memory.get("content", "").lower():
                        matching_memories.append(i)
                if len(matching_memories) >= 3:
                    suggestions["new_category_suggestions"].append({
                        "term": term,
                        "search_count": count,
                        "matching_memories": len(matching_memories),
                        "reason": f"Common search term ({count} searches) with {len(matching_memories)} relevant memories"
                    })

        # 4. Rename suggestions
        for cat_name, utility in self.category_utility.items():
            if utility == 0 and cat_name in self.category_coherence:
                category_content = []
                for memory in self.memory_system.memories:
                    if cat_name in memory.get("categories", []):
                        category_content.append(memory.get("content", ""))
                if category_content:
                    content_words = defaultdict(int)
                    for content in category_content:
                        for word in content.lower().split():
                            if len(word) > 3:
                                content_words[word] += 1
                    potential_names = []
                    for word, frequency in sorted(content_words.items(), key=lambda x: x[1], reverse=True):
                        if word in self.search_patterns and self.search_patterns[word] >= 2:
                            potential_names.append((word, self.search_patterns[word], frequency))
                    if potential_names:
                        best_name = max(potential_names, key=lambda x: x[1] + x[2])
                        suggestions["rename_suggestions"].append({
                            "current_name": cat_name,
                            "suggested_name": best_name[0],
                            "search_frequency": best_name[1],
                            "content_frequency": best_name[2],
                            "reason": f"Category never searched, but '{best_name[0]}' is searched {best_name[1]} times"
                        })

        return suggestions

    def apply_improvements(self, improvement_plan: Dict) -> Dict:
        """Apply suggested improvements to the memory system."""
        results = {"applied_changes": [], "errors": [], "timestamp": datetime.now().isoformat()}

        for merge in improvement_plan.get("merges", []):
            try:
                source_cat, target_cat = merge["source"], merge["target"]
                if source_cat not in self.memory_system.categories or target_cat not in self.memory_system.categories:
                    results["errors"].append(f"Cannot merge: '{source_cat}' or '{target_cat}' not found")
                    continue
                source_memories = [i for i, m in enumerate(self.memory_system.memories) if source_cat in m.get("categories", [])]
                for mid in source_memories:
                    self.memory_system.categorize_memory(mid, [target_cat])
                self.memory_system.add_relationship(source_cat, "merged_into", target_cat, "Merged due to similarity")
                if merge.get("deprecate_source", False):
                    cat = self.memory_system.categories[source_cat]
                    cat["status"] = "deprecated"
                    cat["version"] += 1
                    cat["updated_at"] = datetime.now().isoformat()
                    cat["history"].append({"version": cat["version"], "description": cat["description"],
                                           "status": "deprecated", "reason": f"Merged into {target_cat}",
                                           "timestamp": datetime.now().isoformat()})
                results["applied_changes"].append({"type": "merge", "source": source_cat, "target": target_cat,
                                                    "memories_affected": len(source_memories)})
            except Exception as e:
                results["errors"].append(f"Error merging: {e}")

        for split in improvement_plan.get("splits", []):
            try:
                source_cat, new_cat = split["source"], split["new_category"]
                if source_cat not in self.memory_system.categories:
                    continue
                self.memory_system.add_category(new_cat, split.get("description", f"Split from {source_cat}"))
                self.memory_system.add_relationship(new_cat, "split_from", source_cat, "Split due to low coherence")
                count = 0
                for mid in split.get("memories", []):
                    if 0 <= mid < len(self.memory_system.memories):
                        self.memory_system.categorize_memory(mid, [new_cat])
                        count += 1
                results["applied_changes"].append({"type": "split", "source": source_cat, "new_category": new_cat, "memories_added": count})
            except Exception as e:
                results["errors"].append(f"Error splitting: {e}")

        for new_cat in improvement_plan.get("new_categories", []):
            try:
                cat_name = new_cat["name"]
                if cat_name in self.memory_system.categories:
                    continue
                self.memory_system.add_category(cat_name, new_cat.get("description", "Auto-created"))
                count = 0
                for mid in new_cat.get("memories", []):
                    if 0 <= mid < len(self.memory_system.memories):
                        self.memory_system.categorize_memory(mid, [cat_name])
                        count += 1
                results["applied_changes"].append({"type": "new_category", "name": cat_name, "memories_added": count})
            except Exception as e:
                results["errors"].append(f"Error creating category: {e}")

        for rename in improvement_plan.get("renames", []):
            try:
                old_name, new_name = rename["old_name"], rename["new_name"]
                if old_name not in self.memory_system.categories or new_name in self.memory_system.categories:
                    continue
                old_cat = self.memory_system.categories[old_name]
                self.memory_system.add_category(new_name, rename.get("description", old_cat["description"]))
                count = 0
                for i, m in enumerate(self.memory_system.memories):
                    if old_name in m.get("categories", []):
                        self.memory_system.categorize_memory(i, [new_name])
                        count += 1
                self.memory_system.add_relationship(new_name, "renamed_from", old_name, "Renamed for searchability")
                if rename.get("deprecate_old", False):
                    old_cat["status"] = "deprecated"
                    old_cat["version"] += 1
                    old_cat["updated_at"] = datetime.now().isoformat()
                    old_cat["history"].append({"version": old_cat["version"], "description": old_cat["description"],
                                               "status": "deprecated", "reason": f"Renamed to {new_name}",
                                               "timestamp": datetime.now().isoformat()})
                results["applied_changes"].append({"type": "rename", "old_name": old_name, "new_name": new_name, "memories_updated": count})
            except Exception as e:
                results["errors"].append(f"Error renaming: {e}")

        self.update_quality_metrics()
        return results

    def run_feedback_cycle(self, auto_apply: bool = False) -> Dict:
        metrics = self.update_quality_metrics()
        suggestions = self.suggest_improvements()
        results = {"metrics": metrics, "suggestions": suggestions, "applied_changes": None,
                   "timestamp": datetime.now().isoformat()}

        if auto_apply:
            improvement_plan = {
                "merges": [{"source": m["categories"][0], "target": m["categories"][1], "deprecate_source": False}
                           for m in suggestions["merge_suggestions"]],
                "splits": [],
                "new_categories": [],
                "renames": [{"old_name": r["current_name"], "new_name": r["suggested_name"], "deprecate_old": False}
                            for r in suggestions["rename_suggestions"]]
            }
            results["applied_changes"] = self.apply_improvements(improvement_plan)

        return results
