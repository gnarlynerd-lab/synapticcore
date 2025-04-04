import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

# This extends the existing MemorySystem class with dual embedding spaces
# Only showing the new and modified methods for clarity
class EnhancedMemorySystem:
    """
    Enhancement methods for the MemorySystem class to implement dual embedding spaces.
    These methods can be added to the existing MemorySystem class.
    """
    
    def get_category_embedding(self, category_name: str) -> Optional[np.ndarray]:
        """
        Get aggregate embedding for a category based on its member memories.
        
        Args:
            category_name: Name of the category
            
        Returns:
            Numpy array representing the category's aggregate embedding, or None if not available
        """
        # Check if category exists
        if category_name not in self.categories:
            return None
        
        # Find all memories in this category
        memories = [m for m in self.memories 
                   if category_name in m.get("categories", [])]
        
        if not memories:
            return None
        
        # Extract embeddings
        embeddings = [m["embedding"] for m in memories if "embedding" in m and m["embedding"]]
        
        if not embeddings:
            return None
            
        # Convert to numpy arrays if they're lists
        np_embeddings = [np.array(emb) if isinstance(emb, list) else emb for emb in embeddings]
        
        # Return average embedding (could be weighted by importance/recency in the future)
        return np.mean(np_embeddings, axis=0)
    
    def get_category_distribution(self, category_name: str) -> Optional[Dict]:
        """
        Get statistical distribution of memories in a category.
        
        Args:
            category_name: Name of the category
            
        Returns:
            Dictionary with distribution metrics, or None if category doesn't exist
        """
        if category_name not in self.categories:
            return None
            
        # Find indices of memories in this category
        memory_indices = [i for i, m in enumerate(self.memories) 
                         if category_name in m.get("categories", [])]
        
        # Get category center (average embedding)
        center = self.get_category_embedding(category_name)
        
        # Find memory embeddings for variance calculation
        embeddings = []
        for i in memory_indices:
            memory = self.memories[i]
            if "embedding" in memory and memory["embedding"]:
                emb = memory["embedding"]
                if isinstance(emb, list):
                    emb = np.array(emb)
                embeddings.append(emb)
        
        # Calculate variance if we have embeddings and a center
        variance = None
        if center is not None and embeddings:
            # Calculate average distance from center (simplified variance)
            distances = [np.linalg.norm(emb - center) for emb in embeddings]
            variance = np.mean(distances)
        
        return {
            "center": center,
            "member_count": len(memory_indices),
            "memory_indices": memory_indices,
            "variance": variance
        }
    
    def get_category_similarity(self, category1: str, category2: str) -> Optional[float]:
        """
        Calculate similarity between two categories based on their embeddings.
        
        Args:
            category1: First category name
            category2: Second category name
            
        Returns:
            Cosine similarity between categories, or None if one/both don't exist
        """
        emb1 = self.get_category_embedding(category1)
        emb2 = self.get_category_embedding(category2)
        
        if emb1 is None or emb2 is None:
            return None
        
        # Ensure numpy arrays
        if isinstance(emb1, list):
            emb1 = np.array(emb1)
        if isinstance(emb2, list):
            emb2 = np.array(emb2)
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 > 0 and norm2 > 0:
            return np.dot(emb1, emb2) / (norm1 * norm2)
        return 0.0
    
    def suggest_categories_for_memory(self, memory_id: int, threshold: float = 0.7, 
                                    max_suggestions: int = 3) -> List[Dict]:
        """
        Suggest categories for a memory based on embedding similarity to category centers.
        
        Args:
            memory_id: ID of the memory
            threshold: Minimum similarity threshold for suggestions
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of category suggestions with similarity scores
        """
        if memory_id < 0 or memory_id >= len(self.memories):
            return []
            
        memory = self.memories[memory_id]
        
        # Skip if memory doesn't have embedding
        if "embedding" not in memory or not memory["embedding"]:
            return []
            
        memory_embedding = memory["embedding"]
        if isinstance(memory_embedding, list):
            memory_embedding = np.array(memory_embedding)
            
        # Get current categories
        current_categories = set(memory.get("categories", []))
        
        # Calculate similarity to each category
        suggestions = []
        for cat_name in self.categories:
            # Skip if already categorized
            if cat_name in current_categories:
                continue
                
            # Get category embedding
            cat_embedding = self.get_category_embedding(cat_name)
            if cat_embedding is None:
                continue
                
            if isinstance(cat_embedding, list):
                cat_embedding = np.array(cat_embedding)
                
            # Calculate similarity
            norm_mem = np.linalg.norm(memory_embedding)
            norm_cat = np.linalg.norm(cat_embedding)
            
            if norm_mem > 0 and norm_cat > 0:
                similarity = np.dot(memory_embedding, cat_embedding) / (norm_mem * norm_cat)
                
                # Add if above threshold
                if similarity >= threshold:
                    suggestions.append({
                        "category": cat_name,
                        "similarity": float(similarity),
                        "description": self.categories[cat_name]["description"]
                    })
        
        # Sort by similarity
        suggestions.sort(key=lambda x: x["similarity"], reverse=True)
        
        return suggestions[:max_suggestions]
    
    def discover_category_relationships(self, min_similarity: float = 0.6) -> List[Dict]:
        """
        Discover potential relationships between categories based on embedding similarity.
        
        Args:
            min_similarity: Minimum similarity threshold for relationship suggestions
            
        Returns:
            List of suggested relationships
        """
        active_categories = [name for name, info in self.categories.items() 
                            if info["status"] == "active"]
        
        # Need at least two categories
        if len(active_categories) < 2:
            return []
            
        suggestions = []
        
        # Compare all pairs
        for i, cat1 in enumerate(active_categories):
            for cat2 in active_categories[i+1:]:
                similarity = self.get_category_similarity(cat1, cat2)
                
                if similarity and similarity >= min_similarity:
                    # Determine relationship type based on similarity
                    rel_type = "related_to"
                    if similarity > 0.8:
                        rel_type = "similar_to"
                        
                    suggestions.append({
                        "source": cat1,
                        "target": cat2,
                        "similarity": float(similarity),
                        "suggested_type": rel_type,
                        "description": f"Categories share {similarity:.1%} similarity in embedding space"
                    })
        
        # Sort by similarity
        suggestions.sort(key=lambda x: x["similarity"], reverse=True)
        
        return suggestions
    
    def find_category_outliers(self, category_name: str, threshold: float = 1.5) -> List[Dict]:
        """
        Find memories that are outliers in their assigned category.
        
        Args:
            category_name: Name of the category to check
            threshold: Standard deviation multiplier for outlier detection
            
        Returns:
            List of memories that are potential outliers
        """
        if category_name not in self.categories:
            return []
            
        # Get category distribution
        distribution = self.get_category_distribution(category_name)
        if not distribution or distribution["member_count"] < 3:
            return []
            
        center = distribution["center"]
        if center is None:
            return []
            
        if isinstance(center, list):
            center = np.array(center)
            
        # Threshold based on variance
        variance = distribution["variance"]
        if variance is None:
            return []
            
        distance_threshold = variance * threshold
        
        # Check each memory
        outliers = []
        for i in distribution["memory_indices"]:
            memory = self.memories[i]
            if "embedding" not in memory or not memory["embedding"]:
                continue
                
            emb = memory["embedding"]
            if isinstance(emb, list):
                emb = np.array(emb)
                
            # Calculate distance to center
            distance = np.linalg.norm(emb - center)
            
            # Check if outlier
            if distance > distance_threshold:
                outliers.append({
                    "memory_id": i,
                    "content": memory["content"][:100] + ("..." if len(memory["content"]) > 100 else ""),
                    "distance": float(distance),
                    "threshold": float(distance_threshold)
                })
        
        # Sort by distance (most outlying first)
        outliers.sort(key=lambda x: x["distance"], reverse=True)
        
        return outliers
    
    def suggest_new_categories(self, min_memories: int = 3, 
                             similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Suggest new categories based on clusters of uncategorized memories.
        Uses a simplified clustering approach.
        
        Args:
            min_memories: Minimum number of memories for a new category
            similarity_threshold: Similarity threshold for clustering
            
        Returns:
            List of suggested new categories with sample memories
        """
        # Find uncategorized or poorly categorized memories
        uncategorized = [i for i, m in enumerate(self.memories) 
                        if not m.get("categories") and "embedding" in m and m["embedding"]]
        
        # Not enough uncategorized memories
        if len(uncategorized) < min_memories:
            return []
            
        # Very simple clustering approach
        clusters = []
        used_indices = set()
        
        for i in uncategorized:
            if i in used_indices:
                continue
                
            memory = self.memories[i]
            emb1 = memory["embedding"]
            if isinstance(emb1, list):
                emb1 = np.array(emb1)
                
            # Create new cluster
            cluster = [i]
            used_indices.add(i)
            
            # Find similar memories
            for j in uncategorized:
                if j in used_indices:
                    continue
                    
                mem2 = self.memories[j]
                emb2 = mem2["embedding"]
                if isinstance(emb2, list):
                    emb2 = np.array(emb2)
                    
                # Calculate similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                # Add to cluster if similar
                if similarity >= similarity_threshold:
                    cluster.append(j)
                    used_indices.add(j)
            
            # Keep if enough memories
            if len(cluster) >= min_memories:
                # Calculate cluster center
                cluster_embeddings = [np.array(self.memories[j]["embedding"]) 
                                     if isinstance(self.memories[j]["embedding"], list) 
                                     else self.memories[j]["embedding"] 
                                     for j in cluster]
                center = np.mean(cluster_embeddings, axis=0)
                
                clusters.append({
                    "memory_indices": cluster,
                    "center": center.tolist(),
                    "size": len(cluster),
                    "sample_texts": [self.memories[j]["content"][:100] + "..." 
                                    for j in cluster[:3]]
                })
        
        return clusters

# Integration with existing MemorySystem class
def enhance_memory_system(memory_system):
    """
    Enhance an existing MemorySystem instance with the dual embedding spaces functionality.
    
    Args:
        memory_system: Instance of MemorySystem class
        
    Returns:
        Enhanced MemorySystem with new methods
    """
    # Add new methods
    memory_system.get_category_embedding = EnhancedMemorySystem.get_category_embedding.__get__(memory_system)
    memory_system.get_category_distribution = EnhancedMemorySystem.get_category_distribution.__get__(memory_system)
    memory_system.get_category_similarity = EnhancedMemorySystem.get_category_similarity.__get__(memory_system)
    memory_system.suggest_categories_for_memory = EnhancedMemorySystem.suggest_categories_for_memory.__get__(memory_system)
    memory_system.discover_category_relationships = EnhancedMemorySystem.discover_category_relationships.__get__(memory_system)
    memory_system.find_category_outliers = EnhancedMemorySystem.find_category_outliers.__get__(memory_system)
    memory_system.suggest_new_categories = EnhancedMemorySystem.suggest_new_categories.__get__(memory_system)
    
    return memory_system


# Example usage
def demo_dual_embedding():
    """Demonstrate the dual embedding spaces functionality."""
    from simple_memory_system import MemorySystem
    
    # Initialize memory system
    memory_system = MemorySystem(storage_path="enhanced_memory.json")
    
    # Enhance with dual embedding spaces
    memory_system = enhance_memory_system(memory_system)
    
    # Add some test categories and memories if needed
    if len(memory_system.memories) < 5:
        memory_system.add_category("programming", "Computer programming and software development")
        memory_system.add_category("machine_learning", "Machine learning techniques and applications")
        memory_system.add_category("personal", "Personal experiences and thoughts")
        
        memory_system.add_memory(
            "Python is my favorite programming language because of its readability and extensive libraries.",
            categories=["programming"]
        )
        
        memory_system.add_memory(
            "I learned about neural networks last week, they're very powerful for image classification.",
            categories=["machine_learning"]
        )
        
        memory_system.add_memory(
            "TensorFlow and PyTorch are popular frameworks for building deep learning models.",
            categories=["programming", "machine_learning"]
        )
        
        memory_system.add_memory(
            "I enjoyed hiking in the mountains last weekend, the weather was perfect.",
            categories=["personal"]
        )
        
        memory_system.add_memory(
            "Software design patterns help create maintainable and extensible code.",
            categories=["programming"]
        )
    
    # Demonstrate dual embedding spaces functionality
    print("\nDual Embedding Spaces Demonstration")
    print("=" * 50)
    
    # Show category embeddings
    for cat_name in memory_system.categories:
        emb = memory_system.get_category_embedding(cat_name)
        if emb is not None:
            print(f"Category '{cat_name}' embedding shape: {emb.shape}")
    
    # Show category similarities
    print("\nCategory Similarities:")
    for cat1 in memory_system.categories:
        for cat2 in memory_system.categories:
            if cat1 != cat2:
                similarity = memory_system.get_category_similarity(cat1, cat2)
                if similarity:
                    print(f"  {cat1} <-> {cat2}: {similarity:.3f}")
    
    # Discover potential relationships
    print("\nSuggested Category Relationships:")
    relationships = memory_system.discover_category_relationships(min_similarity=0.5)
    for rel in relationships:
        print(f"  {rel['source']} {rel['suggested_type']} {rel['target']} ({rel['similarity']:.3f})")
    
    # Add a memory and suggest categories
    if memory_system.embedding_model:
        print("\nSuggesting Categories for New Memory:")
        memory_id = memory_system.add_memory(
            "Deep learning models require large datasets for training to achieve good performance."
        )
        
        suggestions = memory_system.suggest_categories_for_memory(memory_id, threshold=0.6)
        print(f"  Memory: {memory_system.memories[memory_id]['content']}")
        print("  Suggested categories:")
        for suggestion in suggestions:
            print(f"    - {suggestion['category']} ({suggestion['similarity']:.3f})")
    
    # Find outliers in categories
    print("\nOutlier Detection:")
    for cat_name in memory_system.categories:
        outliers = memory_system.find_category_outliers(cat_name)
        if outliers:
            print(f"  Outliers in '{cat_name}':")
            for outlier in outliers:
                print(f"    - {outlier['content']} (distance: {outlier['distance']:.3f})")
    
    return memory_system

if __name__ == "__main__":
    demo_dual_embedding()