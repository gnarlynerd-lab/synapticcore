"""
Enhanced memory with dual embedding spaces.

Refactored from enhanced_memory_system.py — composition instead of monkey-patching.
Wraps a MemorySystem and adds category-level embedding operations.
"""

import numpy as np
from typing import List, Dict, Optional

from .base import MemorySystem


class EnhancedMemory:
    """
    Dual embedding space operations on top of a MemorySystem.

    Provides category-level embeddings (aggregated from member memories),
    similarity scoring, outlier detection, and category suggestions.
    """

    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system

    # Convenience accessors
    @property
    def memories(self):
        return self.memory_system.memories

    @property
    def categories(self):
        return self.memory_system.categories

    def get_category_embedding(self, category_name: str) -> Optional[np.ndarray]:
        """Get aggregate embedding for a category (mean of member embeddings)."""
        if category_name not in self.categories:
            return None
        memories = [m for m in self.memories if category_name in m.get("categories", [])]
        if not memories:
            return None
        embeddings = [m["embedding"] for m in memories if m.get("embedding")]
        if not embeddings:
            return None
        np_embeddings = [np.array(emb) if isinstance(emb, list) else emb for emb in embeddings]
        return np.mean(np_embeddings, axis=0)

    def get_category_distribution(self, category_name: str) -> Optional[Dict]:
        """Get statistical distribution of memories in a category."""
        if category_name not in self.categories:
            return None
        memory_indices = [i for i, m in enumerate(self.memories) if category_name in m.get("categories", [])]
        center = self.get_category_embedding(category_name)

        embeddings = []
        for i in memory_indices:
            memory = self.memories[i]
            if memory.get("embedding"):
                emb = memory["embedding"]
                if isinstance(emb, list):
                    emb = np.array(emb)
                embeddings.append(emb)

        variance = None
        if center is not None and embeddings:
            distances = [np.linalg.norm(emb - center) for emb in embeddings]
            variance = float(np.mean(distances))

        return {
            "center": center,
            "member_count": len(memory_indices),
            "memory_indices": memory_indices,
            "variance": variance
        }

    def get_category_similarity(self, category1: str, category2: str) -> Optional[float]:
        """Cosine similarity between two categories."""
        emb1 = self.get_category_embedding(category1)
        emb2 = self.get_category_embedding(category2)
        if emb1 is None or emb2 is None:
            return None
        if isinstance(emb1, list):
            emb1 = np.array(emb1)
        if isinstance(emb2, list):
            emb2 = np.array(emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 > 0 and norm2 > 0:
            return float(np.dot(emb1, emb2) / (norm1 * norm2))
        return 0.0

    def suggest_categories_for_memory(self, memory_id: int, threshold: float = 0.7,
                                      max_suggestions: int = 3) -> List[Dict]:
        """Suggest categories for a memory based on embedding similarity to category centers."""
        if memory_id < 0 or memory_id >= len(self.memories):
            return []
        memory = self.memories[memory_id]
        if not memory.get("embedding"):
            return []

        memory_embedding = memory["embedding"]
        if isinstance(memory_embedding, list):
            memory_embedding = np.array(memory_embedding)

        current_categories = set(memory.get("categories", []))
        suggestions = []

        for cat_name in self.categories:
            if cat_name in current_categories:
                continue
            cat_embedding = self.get_category_embedding(cat_name)
            if cat_embedding is None:
                continue
            if isinstance(cat_embedding, list):
                cat_embedding = np.array(cat_embedding)

            norm_mem = np.linalg.norm(memory_embedding)
            norm_cat = np.linalg.norm(cat_embedding)
            if norm_mem > 0 and norm_cat > 0:
                similarity = float(np.dot(memory_embedding, cat_embedding) / (norm_mem * norm_cat))
                if similarity >= threshold:
                    suggestions.append({
                        "category": cat_name,
                        "similarity": similarity,
                        "description": self.categories[cat_name]["description"]
                    })

        suggestions.sort(key=lambda x: x["similarity"], reverse=True)
        return suggestions[:max_suggestions]

    def discover_category_relationships(self, min_similarity: float = 0.6) -> List[Dict]:
        """Discover potential relationships between categories based on embedding similarity."""
        active_categories = [name for name, info in self.categories.items() if info["status"] == "active"]
        if len(active_categories) < 2:
            return []

        suggestions = []
        for i, cat1 in enumerate(active_categories):
            for cat2 in active_categories[i + 1:]:
                similarity = self.get_category_similarity(cat1, cat2)
                if similarity and similarity >= min_similarity:
                    rel_type = "similar_to" if similarity > 0.8 else "related_to"
                    suggestions.append({
                        "source": cat1,
                        "target": cat2,
                        "similarity": similarity,
                        "suggested_type": rel_type,
                        "description": f"Categories share {similarity:.1%} similarity in embedding space"
                    })

        suggestions.sort(key=lambda x: x["similarity"], reverse=True)
        return suggestions

    def find_category_outliers(self, category_name: str, threshold: float = 1.5) -> List[Dict]:
        """Find memories that are outliers in their assigned category."""
        if category_name not in self.categories:
            return []
        distribution = self.get_category_distribution(category_name)
        if not distribution or distribution["member_count"] < 3:
            return []

        center = distribution["center"]
        if center is None:
            return []
        if isinstance(center, list):
            center = np.array(center)

        variance = distribution["variance"]
        if variance is None:
            return []
        distance_threshold = variance * threshold

        outliers = []
        for i in distribution["memory_indices"]:
            memory = self.memories[i]
            if not memory.get("embedding"):
                continue
            emb = memory["embedding"]
            if isinstance(emb, list):
                emb = np.array(emb)
            distance = float(np.linalg.norm(emb - center))
            if distance > distance_threshold:
                outliers.append({
                    "memory_id": i,
                    "content": memory["content"][:100] + ("..." if len(memory["content"]) > 100 else ""),
                    "distance": distance,
                    "threshold": float(distance_threshold)
                })

        outliers.sort(key=lambda x: x["distance"], reverse=True)
        return outliers

    def suggest_new_categories(self, min_memories: int = 3,
                               similarity_threshold: float = 0.7) -> List[Dict]:
        """Suggest new categories based on clusters of uncategorized memories."""
        uncategorized = [i for i, m in enumerate(self.memories)
                         if not m.get("categories") and m.get("embedding")]
        if len(uncategorized) < min_memories:
            return []

        clusters = []
        used_indices = set()

        for i in uncategorized:
            if i in used_indices:
                continue
            memory = self.memories[i]
            emb1 = memory["embedding"]
            if isinstance(emb1, list):
                emb1 = np.array(emb1)

            cluster = [i]
            used_indices.add(i)

            for j in uncategorized:
                if j in used_indices:
                    continue
                mem2 = self.memories[j]
                emb2 = mem2["embedding"]
                if isinstance(emb2, list):
                    emb2 = np.array(emb2)
                similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                if similarity >= similarity_threshold:
                    cluster.append(j)
                    used_indices.add(j)

            if len(cluster) >= min_memories:
                cluster_embeddings = [
                    np.array(self.memories[j]["embedding"]) if isinstance(self.memories[j]["embedding"], list)
                    else self.memories[j]["embedding"]
                    for j in cluster
                ]
                center = np.mean(cluster_embeddings, axis=0)
                clusters.append({
                    "memory_indices": cluster,
                    "center": center.tolist(),
                    "size": len(cluster),
                    "sample_texts": [self.memories[j]["content"][:100] + "..." for j in cluster[:3]]
                })

        return clusters
