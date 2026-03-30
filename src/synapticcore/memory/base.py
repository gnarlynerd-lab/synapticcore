"""
Core Memory System with embeddings and category evolution.

Refactored from simple_memory_system.py — same behavior, uses StorageBackend
instead of direct file I/O, and delegates search to HybridSearchEngine.
"""

from datetime import datetime
import logging
import numpy as np
import uuid
import random
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import hnswlib
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from ..storage.base import StorageBackend
from ..storage.json_store import JsonFileStore


class MemorySystem:
    """
    Core memory system with embedding-based retrieval and evolving categories.
    """

    def __init__(self, storage=None, storage_path: str = "memory_store.json",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Args:
            storage: StorageBackend instance. If None, creates a JsonFileStore.
            storage_path: Used only if storage is None.
            embedding_model: Sentence transformer model name.
        """
        if storage is None:
            storage = JsonFileStore(storage_path)
        self.storage = storage

        self.memories: List[Dict] = []
        self.categories: Dict[str, Dict] = {}
        self.relationships: Dict[str, Dict] = {}

        # Embedding infrastructure
        self.embedding_model = None
        self.vector_index = None
        self.memory_to_index: Dict[int, int] = {}
        self.index_to_memory: Dict[int, int] = {}

        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")

        # Load existing data
        self._load_data()

        # Initialize vector index
        if EMBEDDINGS_AVAILABLE and self.embedding_model and not self.vector_index:
            self._initialize_vector_index()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load_data(self):
        data = self.storage.load()
        self.memories = data.get("memories", [])
        self.categories = data.get("categories", {})
        self.relationships = data.get("relationships", {})
        if self.memories:
            logger.info(f"Loaded {len(self.memories)} memories and {len(self.categories)} categories")

    def _save_data(self):
        self.storage.save({
            "memories": self.memories,
            "categories": self.categories,
            "relationships": self.relationships,
        })

    def _initialize_vector_index(self):
        """Initialize or load the HNSW vector index."""
        if not self.memories:
            sample_embedding = self.embedding_model.encode("Sample text to determine dimension.")
        else:
            sample_memory = next((m for m in self.memories if m.get("content")), None)
            if sample_memory:
                sample_embedding = self.embedding_model.encode(sample_memory["content"])
            else:
                sample_embedding = self.embedding_model.encode("Sample text")

        dim = len(sample_embedding)

        # Try loading persisted index
        loaded = self.storage.load_index(dim)
        if loaded is not None:
            index, mem_to_idx, idx_to_mem = loaded
            # Validate mapping
            if not mem_to_idx or max(mem_to_idx.keys()) < len(self.memories):
                self.vector_index = index
                self.memory_to_index = mem_to_idx
                self.index_to_memory = idx_to_mem
                logger.info(f"Loaded HNSW index with {self.vector_index.get_current_count()} vectors")
                return

        # Build fresh index
        self.vector_index = hnswlib.Index(space='cosine', dim=dim)
        self.vector_index.init_index(
            max_elements=max(1000, len(self.memories) * 2),
            ef_construction=200, M=16
        )
        self.vector_index.set_ef(50)

        count = 0
        for i, memory in enumerate(self.memories):
            if memory.get("embedding"):
                self._add_to_index(i, memory["embedding"])
                count += 1

        logger.info(f"Built vector index with {count} memories")
        self.storage.save_index(self.vector_index, self.memory_to_index, self.index_to_memory)

    def _add_to_index(self, memory_id: int, embedding):
        """Add a memory embedding to the HNSW index."""
        if not self.vector_index:
            return
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        index_id = len(self.memory_to_index)
        self.vector_index.add_items(embedding, index_id)
        self.memory_to_index[memory_id] = index_id
        self.index_to_memory[index_id] = memory_id

    # ── Memory CRUD ──────────────────────────────────────────────────────

    def add_memory(self, content: str, categories: List[str] = None,
                   metadata: Dict = None) -> int:
        """Add a new memory. Returns its integer ID."""
        embedding = None
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(content).tolist()
            except Exception as e:
                logger.error(f"Error creating embedding: {e}")

        memory = {
            "id": str(uuid.uuid4()),
            "content": content,
            "categories": categories or [],
            "embedding": embedding,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        memory_id = len(self.memories)
        self.memories.append(memory)

        if embedding and self.vector_index:
            self._add_to_index(memory_id, embedding)
            self.storage.save_index(self.vector_index, self.memory_to_index, self.index_to_memory)

        if categories:
            for category in categories:
                if category not in self.categories:
                    self.add_category(category)

        self._save_data()
        return memory_id

    def add_category(self, name: str, description: str = "") -> bool:
        if name in self.categories:
            return False
        self.categories[name] = {
            "name": name,
            "description": description,
            "version": 1,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "history": [{
                "version": 1,
                "description": description,
                "status": "active",
                "timestamp": datetime.now().isoformat()
            }]
        }
        self._save_data()
        return True

    def update_category(self, name: str, description: str) -> bool:
        if name not in self.categories:
            return False
        category = self.categories[name]
        if description != category["description"]:
            category["version"] += 1
            category["description"] = description
            category["updated_at"] = datetime.now().isoformat()
            category["history"].append({
                "version": category["version"],
                "description": description,
                "status": category["status"],
                "timestamp": datetime.now().isoformat()
            })
            self._save_data()
        return True

    def categorize_memory(self, memory_id: int, categories: List[str]) -> bool:
        if memory_id < 0 or memory_id >= len(self.memories):
            return False
        memory = self.memories[memory_id]
        old_categories = memory.get("categories", [])
        memory["categories"] = list(set(old_categories + categories))

        if "category_history" not in memory:
            memory["category_history"] = []
        if set(old_categories) != set(memory["categories"]):
            memory["category_history"].append({
                "timestamp": datetime.now().isoformat(),
                "old": old_categories,
                "new": memory["categories"]
            })

        for category in categories:
            if category not in self.categories:
                self.add_category(category)

        self._save_data()
        return True

    def add_relationship(self, category1: str, relationship_type: str,
                         category2: str, description: str = "") -> bool:
        if category1 not in self.categories or category2 not in self.categories:
            return False
        rel_id = f"{category1}:{relationship_type}:{category2}"

        if rel_id in self.relationships:
            relationship = self.relationships[rel_id]
            relationship["version"] += 1
            relationship["description"] = description
            relationship["updated_at"] = datetime.now().isoformat()
            relationship["history"].append({
                "version": relationship["version"],
                "description": description,
                "status": relationship["status"],
                "timestamp": datetime.now().isoformat()
            })
        else:
            self.relationships[rel_id] = {
                "source": category1,
                "target": category2,
                "type": relationship_type,
                "description": description,
                "version": 1,
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "history": [{
                    "version": 1,
                    "description": description,
                    "status": "active",
                    "timestamp": datetime.now().isoformat()
                }]
            }

        self._save_data()
        return True

    # ── Search ───────────────────────────────────────────────────────────

    def retrieve_by_similarity(self, query: str, top_k: int = 5,
                               category_filter: List[str] = None) -> List[Dict]:
        if not self.embedding_model or not self.vector_index:
            return []
        query_embedding = self.embedding_model.encode(query)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        count = self.vector_index.get_current_count()
        if count == 0:
            return []
        labels, distances = self.vector_index.knn_query(query_embedding, k=min(top_k * 3, count))

        results = []
        for idx, dist in zip(labels[0], distances[0]):
            similarity = float(1.0 - dist)
            memory_id = self.index_to_memory.get(int(idx))
            if memory_id is None:
                continue
            memory = self.memories[memory_id]
            if category_filter and not any(cat in memory.get("categories", []) for cat in category_filter):
                continue
            results.append({
                "memory": memory,
                "similarity": similarity,
                "memory_id": memory_id
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def retrieve_by_category(self, categories: List[str], limit: int = 10) -> List[Dict]:
        results = []
        for i, memory in enumerate(self.memories):
            memory_categories = memory.get("categories", [])
            if any(cat in memory_categories for cat in categories):
                results.append({
                    "memory": memory,
                    "memory_id": i,
                    "categories": [cat for cat in categories if cat in memory_categories]
                })
        results.sort(key=lambda x: x["memory"].get("timestamp", ""), reverse=True)
        return results[:limit]

    def hybrid_search(self, query: str, categories: List[str] = None,
                      top_k: int = 5, category_weight: float = 0.3) -> List[Dict]:
        similarity_results = self.retrieve_by_similarity(query, top_k=top_k * 2)
        if not categories:
            return similarity_results[:top_k]

        combined_results = []
        for result in similarity_results:
            memory = result["memory"]
            memory_categories = memory.get("categories", [])
            base_score = result["similarity"]
            category_boost = 0.0
            if categories:
                matches = sum(1 for cat in categories if cat in memory_categories)
                if matches > 0:
                    category_boost = matches / max(len(categories), 1) * category_weight
            combined_score = base_score * (1 - category_weight) + category_boost
            combined_results.append({
                "memory": memory,
                "memory_id": result["memory_id"],
                "similarity": base_score,
                "category_boost": category_boost,
                "combined_score": combined_score
            })

        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined_results[:top_k]

    def enhanced_hybrid_search(self, query: str, categories: List[str] = None,
                               top_k: int = 5, semantic_weight: float = 0.5,
                               recursive_depth: int = 1) -> List[Dict]:
        """4-phase search: semantic, category, recency, associative expansion."""
        import math
        results = []

        # Phase 1: Semantic similarity
        if self.embedding_model and self.vector_index:
            semantic_results = self.retrieve_by_similarity(query, top_k=top_k * 2)
            for result in semantic_results:
                results.append({
                    "memory": result["memory"],
                    "memory_id": result["memory_id"],
                    "semantic_score": result["similarity"],
                    "category_score": 0.0,
                    "recency_score": 0.0,
                    "combined_score": result["similarity"] * semantic_weight,
                    "retrieval_method": "semantic"
                })

        # Phase 2: Category-based retrieval
        category_weight = 1.0 - semantic_weight
        if category_weight > 0:
            search_categories = categories if categories else self._infer_categories_from_query(query)
            if search_categories:
                category_results = self.retrieve_by_category(search_categories, limit=top_k * 2)
                for result in category_results:
                    memory = result["memory"]
                    memory_id = next((i for i, mem in enumerate(self.memories) if mem == memory), None)
                    if memory_id is None:
                        continue
                    memory_categories = set(memory.get("categories", []))
                    search_cat_set = set(search_categories)
                    category_match = len(memory_categories.intersection(search_cat_set)) / max(1, len(search_cat_set))

                    existing_idx = next((i for i, r in enumerate(results) if r.get("memory_id") == memory_id), None)
                    if existing_idx is not None:
                        results[existing_idx]["category_score"] = category_match
                        results[existing_idx]["combined_score"] += category_match * category_weight
                        results[existing_idx]["retrieval_method"] += "+category"
                    else:
                        semantic_score = 0.0
                        if self.embedding_model and memory.get("embedding"):
                            query_embedding = self.embedding_model.encode(query)
                            memory_embedding = memory["embedding"]
                            if isinstance(memory_embedding, list):
                                memory_embedding = np.array(memory_embedding)
                            query_norm = np.linalg.norm(query_embedding)
                            memory_norm = np.linalg.norm(memory_embedding)
                            if query_norm > 0 and memory_norm > 0:
                                semantic_score = float(np.dot(query_embedding, memory_embedding) / (query_norm * memory_norm))

                        results.append({
                            "memory": memory,
                            "memory_id": memory_id,
                            "semantic_score": semantic_score,
                            "category_score": category_match,
                            "recency_score": 0.0,
                            "combined_score": (semantic_score * semantic_weight) + (category_match * category_weight),
                            "retrieval_method": "category"
                        })

        # Phase 3: Recency boost
        if self.memories:
            latest_timestamp = max((m.get("timestamp", "") for m in self.memories if "timestamp" in m), default="")
            if latest_timestamp:
                try:
                    latest_time = datetime.fromisoformat(latest_timestamp)
                    for result in results:
                        memory = result["memory"]
                        if "timestamp" in memory:
                            memory_time = datetime.fromisoformat(memory["timestamp"])
                            time_diff = (latest_time - memory_time).total_seconds()
                            half_life = 60 * 60 * 24 * 7  # one week
                            recency_score = math.exp(-math.log(2) * time_diff / half_life)
                            result["recency_score"] = recency_score
                            result["combined_score"] += 0.1 * recency_score
                except (ValueError, TypeError):
                    pass

        # Phase 4: Associative expansion
        if recursive_depth > 0 and results:
            top_results = sorted(results, key=lambda x: x["combined_score"], reverse=True)[:3]
            for top_result in top_results:
                memory = top_result["memory"]
                shared_categories = memory.get("categories", [])
                if shared_categories:
                    associated_results = self.retrieve_by_category(shared_categories, limit=2 * recursive_depth)
                    for assoc in associated_results:
                        assoc_memory = assoc["memory"]
                        assoc_id = next((i for i, mem in enumerate(self.memories) if mem == assoc_memory), None)
                        if assoc_id is None or assoc_id == top_result["memory_id"] or any(
                            r["memory_id"] == assoc_id for r in results
                        ):
                            continue

                        semantic_score = 0.0
                        if self.embedding_model and assoc_memory.get("embedding"):
                            try:
                                query_embedding = self.embedding_model.encode(query)
                                memory_embedding = assoc_memory["embedding"]
                                if isinstance(memory_embedding, list):
                                    memory_embedding = np.array(memory_embedding)
                                semantic_score = float(
                                    np.dot(query_embedding, memory_embedding) /
                                    (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding))
                                )
                            except Exception:
                                pass

                        memory_cats = set(memory.get("categories", []))
                        assoc_cats = set(assoc_memory.get("categories", []))
                        association_strength = len(memory_cats.intersection(assoc_cats)) / max(1, len(memory_cats))
                        association_score = association_strength * (0.3 / recursive_depth)

                        results.append({
                            "memory": assoc_memory,
                            "memory_id": assoc_id,
                            "semantic_score": semantic_score,
                            "category_score": 0,
                            "association_score": association_score,
                            "combined_score": semantic_score * 0.3 + association_score,
                            "retrieval_method": "association",
                            "associated_with": top_result["memory_id"]
                        })

                if recursive_depth > 1:
                    memory_content = memory.get("content", "")[:200]
                    recursive_results = self.enhanced_hybrid_search(
                        memory_content,
                        categories=memory.get("categories", []),
                        top_k=recursive_depth,
                        semantic_weight=semantic_weight,
                        recursive_depth=recursive_depth - 1
                    )
                    for rec_result in recursive_results:
                        rec_id = rec_result["memory_id"]
                        if any(r["memory_id"] == rec_id for r in results):
                            continue
                        rec_result["combined_score"] *= 0.5 / recursive_depth
                        rec_result["retrieval_method"] = "recursive_association"
                        rec_result["associated_with"] = top_result["memory_id"]
                        results.append(rec_result)

        # Deduplicate and sort
        unique_results = {}
        for result in results:
            memory_id = result["memory_id"]
            if memory_id not in unique_results or result["combined_score"] > unique_results[memory_id]["combined_score"]:
                unique_results[memory_id] = result
        final_results = sorted(unique_results.values(), key=lambda x: x["combined_score"], reverse=True)
        return final_results[:top_k]

    def _infer_categories_from_query(self, query: str) -> List[str]:
        """Infer relevant categories from query text."""
        relevant_categories = []
        query_lower = query.lower()

        # Check for category name mentions
        for category_name in self.categories:
            if category_name.lower() in query_lower:
                relevant_categories.append(category_name)

        # Semantic similarity to category embeddings
        if not relevant_categories and self.embedding_model:
            try:
                query_embedding = self.embedding_model.encode(query)
                category_similarities = []
                for category_name, category_info in self.categories.items():
                    if category_info.get("status") != "active":
                        continue
                    # Need EnhancedMemory for category embeddings, skip if not available
                category_similarities.sort(key=lambda x: x[1], reverse=True)
                relevant_categories = [cat for cat, sim in category_similarities if sim > 0.5][:3]
            except Exception:
                pass

        # Fallback: most used categories
        if not relevant_categories:
            category_counts = {}
            for memory in self.memories:
                for category in memory.get("categories", []):
                    category_counts[category] = category_counts.get(category, 0) + 1
            if category_counts:
                sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                relevant_categories = [cat for cat, _ in sorted_cats[:3]]

        return relevant_categories

    # ── Analysis ─────────────────────────────────────────────────────────

    def get_category_evolution(self, category_name: str) -> Optional[Dict]:
        if category_name not in self.categories:
            return None
        category = self.categories[category_name]

        related = []
        for rel_id, rel in self.relationships.items():
            if rel["source"] == category_name:
                related.append({"category": rel["target"], "relationship": rel["type"], "description": rel["description"]})
            elif rel["target"] == category_name:
                related.append({"category": rel["source"], "relationship": f"inverse of {rel['type']}", "description": rel["description"]})

        memory_examples = []
        for i, memory in enumerate(self.memories):
            if category_name in memory.get("categories", []):
                memory_examples.append({
                    "id": i,
                    "content": memory["content"][:100] + ("..." if len(memory["content"]) > 100 else ""),
                    "timestamp": memory["timestamp"]
                })
                if len(memory_examples) >= 5:
                    break

        return {
            "name": category_name,
            "current": {"description": category["description"], "status": category["status"], "version": category["version"]},
            "history": category["history"],
            "related_categories": related,
            "memory_examples": memory_examples
        }

    def generate_statistics(self) -> Dict:
        category_usage = defaultdict(int)
        for memory in self.memories:
            for category in memory.get("categories", []):
                category_usage[category] += 1

        active_categories = sum(1 for c in self.categories.values() if c["status"] == "active")
        deprecated_categories = sum(1 for c in self.categories.values() if c["status"] == "deprecated")
        category_versions = {name: info["version"] for name, info in self.categories.items()}
        avg_version = sum(category_versions.values()) / max(1, len(category_versions))
        relationship_types = defaultdict(int)
        for rel in self.relationships.values():
            relationship_types[rel["type"]] += 1

        return {
            "total_memories": len(self.memories),
            "memories_with_embeddings": sum(1 for m in self.memories if m.get("embedding")),
            "total_categories": len(self.categories),
            "active_categories": active_categories,
            "deprecated_categories": deprecated_categories,
            "total_relationships": len(self.relationships),
            "category_usage": dict(sorted(category_usage.items(), key=lambda x: x[1], reverse=True)[:10]),
            "average_category_version": avg_version,
            "relationship_types": dict(relationship_types)
        }

    def periodic_review(self, llm_client=None):
        """Review and suggest category improvements using an LLM."""
        if not llm_client or len(self.memories) < 10:
            return
        import json as _json
        import re

        recent_memories = self.memories[-20:]
        memory_sample = random.sample(recent_memories, min(10, len(recent_memories)))
        active_categories = {name: info for name, info in self.categories.items() if info["status"] == "active"}

        categories_text = "\n".join([
            f"- {name}: {info['description']} (v{info['version']})"
            for name, info in active_categories.items()
        ])
        memories_text = "\n".join([
            f"Memory: {m['content'][:100]}...\nCategories: {m.get('categories', [])}"
            for m in memory_sample
        ])

        prompt = f"""
        Please analyze my memory categories and suggest improvements:

        CURRENT CATEGORIES:
        {categories_text}

        SAMPLE MEMORIES:
        {memories_text}

        Please suggest:
        1. New categories that would be useful
        2. Categories that should be updated with better descriptions
        3. Categories that should be merged or deprecated
        4. Relationships between categories

        Format your response as JSON:
        {{
            "new_categories": {{"name": "description"}},
            "update_categories": {{"name": "improved description"}},
            "merge_suggestions": [["category1", "category2", "reason"]],
            "deprecate_categories": [["name", "reason"]],
            "relationships": [["category1", "relationship_type", "category2", "description"]]
        }}
        """

        try:
            response = llm_client.generate(prompt)
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                suggestions = _json.loads(json_match.group(1))
                self._process_category_suggestions(suggestions)
        except Exception as e:
            logger.error(f"Error in periodic review: {e}")

    def _process_category_suggestions(self, suggestions):
        for name, description in suggestions.get("new_categories", {}).items():
            if name not in self.categories:
                self.add_category(name, description)
        for name, description in suggestions.get("update_categories", {}).items():
            if name in self.categories:
                self.update_category(name, description)
        for merge in suggestions.get("merge_suggestions", []):
            if len(merge) >= 3:
                cat1, cat2, reason = merge
                if cat1 in self.categories and cat2 in self.categories:
                    self.add_relationship(cat1, "similar_to", cat2, reason)
        for deprecate in suggestions.get("deprecate_categories", []):
            if len(deprecate) >= 2:
                name, reason = deprecate
                if name in self.categories and self.categories[name]["status"] == "active":
                    category = self.categories[name]
                    category["status"] = "deprecated"
                    category["version"] += 1
                    category["updated_at"] = datetime.now().isoformat()
                    category["history"].append({
                        "version": category["version"],
                        "description": category["description"],
                        "status": "deprecated",
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    })
        for rel in suggestions.get("relationships", []):
            if len(rel) >= 4:
                cat1, rel_type, cat2, description = rel
                self.add_relationship(cat1, rel_type, cat2, description)
        self._save_data()
