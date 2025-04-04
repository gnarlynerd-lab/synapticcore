#!/usr/bin/env python3
"""
Simplified Memory System with Embeddings and Category Evolution

This module implements a memory system focused on emergent properties from 
simple storage and retrieval loops, with versioned categories.
"""

from datetime import datetime
import numpy as np
import json
import os
import uuid
import random
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict

# Try to import embedding dependencies
try:
    from sentence_transformers import SentenceTransformer
    import hnswlib
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: Required packages not available. Install with: pip install sentence-transformers hnswlib")


class MemorySystem:
    """
    A simplified memory system with embedding-based retrieval and evolving categories.
    Focuses on simplicity and emergent properties from storage and retrieval loops.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2", 
                 storage_path: str = "memory_store.json"):
        """
        Initialize the memory system.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            storage_path: Path to store memory data
        """
        self.storage_path = storage_path
        self.memories = []
        self.categories = {}  # name -> category info
        self.relationships = {}  # (cat1, rel_type, cat2) -> relationship info
        
        # Initialize embedding functionality
        self.embedding_model = None
        self.vector_index = None
        self.memory_to_index = {}  # Maps memory ID to vector index
        self.index_to_memory = {}  # Maps vector index to memory ID
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"Loaded embedding model: {embedding_model}")
                # Create index later after we know the embedding dimension
            except Exception as e:
                print(f"Error loading embedding model: {e}")
        
        # Load existing data if available
        self._load_data()
        
        # Initialize vector index if we have embeddings
        if EMBEDDINGS_AVAILABLE and self.embedding_model and not self.vector_index:
            self._initialize_vector_index()
    
    def _initialize_vector_index(self):
        """Initialize the vector index for similarity search."""
        # Need a sample embedding to get dimensions
        if not self.memories:
            sample_text = "Sample text to determine embedding dimension."
            sample_embedding = self.embedding_model.encode(sample_text)
        else:
            # Find first memory with content
            sample_memory = next((m for m in self.memories if m["content"]), None)
            if sample_memory:
                sample_embedding = self.embedding_model.encode(sample_memory["content"])
            else:
                sample_embedding = self.embedding_model.encode("Sample text")
        
        dim = len(sample_embedding)
        
        # Initialize HNSW index
        self.vector_index = hnswlib.Index(space='cosine', dim=dim)
        self.vector_index.init_index(max_elements=max(1000, len(self.memories) * 2), 
                                    ef_construction=200, M=16)
        self.vector_index.set_ef(50)  # For search
        
        # Add existing memories to index
        count = 0
        for i, memory in enumerate(self.memories):
            if "embedding" in memory and memory["embedding"]:
                self._add_to_index(i, memory["embedding"])
                count += 1
        
        print(f"Added {count} memories to vector index")
    
    def _add_to_index(self, memory_id: int, embedding: List[float]):
        """Add a memory embedding to the vector index."""
        if not self.vector_index:
            return
            
        # Convert to numpy array if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        # Normalize the embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Add to index
        index_id = len(self.memory_to_index)
        self.vector_index.add_items(embedding, index_id)
        self.memory_to_index[memory_id] = index_id
        self.index_to_memory[index_id] = memory_id
    
    def _load_data(self):
        """Load memories and categories from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                self.memories = data.get("memories", [])
                self.categories = data.get("categories", {})
                self.relationships = data.get("relationships", {})
                
                print(f"Loaded {len(self.memories)} memories and {len(self.categories)} categories")
            except Exception as e:
                print(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save memories and categories to storage."""
        try:
            data = {
                "memories": self.memories,
                "categories": self.categories,
                "relationships": self.relationships
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def add_memory(self, content: str, categories: List[str] = None, 
                 metadata: Dict = None) -> int:
        """
        Add a new memory with optional categories and metadata.
        
        Args:
            content: Text content of the memory
            categories: Initial categories to assign (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            ID of the new memory
        """
        # Create embedding if model is available
        embedding = None
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(content).tolist()
            except Exception as e:
                print(f"Error creating embedding: {e}")
        
        # Create new memory object
        memory = {
            "id": str(uuid.uuid4()),
            "content": content,
            "categories": categories or [],
            "embedding": embedding,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to memories list
        memory_id = len(self.memories)
        self.memories.append(memory)
        
        # Add to vector index if embedding was created
        if embedding and self.vector_index:
            self._add_to_index(memory_id, embedding)
        
        # Create any new categories that don't exist yet
        if categories:
            for category in categories:
                if category not in self.categories:
                    self.add_category(category)
        
        # Save data
        self._save_data()
        
        return memory_id
    
    def add_category(self, name: str, description: str = "") -> bool:
        """
        Add a new category with optional description.
        
        Args:
            name: Category name
            description: Category description (optional)
            
        Returns:
            True if successful, False if category already exists
        """
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
        """
        Update a category description with version tracking.
        
        Args:
            name: Category name
            description: New description
            
        Returns:
            True if successful, False if category doesn't exist
        """
        if name not in self.categories:
            return False
            
        category = self.categories[name]
        
        # Only update if description has changed
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
        """
        Add categories to a memory, creating new categories if needed.
        
        Args:
            memory_id: ID of the memory
            categories: List of category names
            
        Returns:
            True if successful, False if memory doesn't exist
        """
        if memory_id < 0 or memory_id >= len(self.memories):
            return False
            
        memory = self.memories[memory_id]
        
        # Track history of categorization changes
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
        
        # Create any new categories that don't exist yet
        for category in categories:
            if category not in self.categories:
                self.add_category(category)
                
        self._save_data()
        return True
    
    def add_relationship(self, category1: str, relationship_type: str, 
                       category2: str, description: str = "") -> bool:
        """
        Add a relationship between two categories.
        
        Args:
            category1: First category name
            relationship_type: Type of relationship (e.g., "includes", "contradicts")
            category2: Second category name
            description: Description of the relationship
            
        Returns:
            True if successful, False if categories don't exist
        """
        if category1 not in self.categories or category2 not in self.categories:
            return False
            
        rel_id = f"{category1}:{relationship_type}:{category2}"
        
        if rel_id in self.relationships:
            # Update existing relationship
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
            # Create new relationship
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
    
    def retrieve_by_similarity(self, query: str, top_k: int = 5, 
                             category_filter: List[str] = None) -> List[Dict]:
        """
        Retrieve memories by semantic similarity, optionally filtered by categories.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            category_filter: Optional list of categories to filter by
            
        Returns:
            List of memories with similarity scores
        """
        if not self.embedding_model or not self.vector_index:
            return []
            
        # Create query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Normalize embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
            
        # Search index
        labels, distances = self.vector_index.knn_query(query_embedding, k=min(top_k * 3, self.vector_index.get_current_count()))
        
        # Convert to similarities and map to memory IDs
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            similarity = 1.0 - dist
            memory_id = self.index_to_memory.get(int(idx))
            
            if memory_id is not None:
                memory = self.memories[memory_id]
                
                # Apply category filter if provided
                if category_filter and not any(cat in memory.get("categories", []) for cat in category_filter):
                    continue
                    
                results.append({
                    "memory": memory,
                    "similarity": similarity,
                    "memory_id": memory_id
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k]
    
    def retrieve_by_category(self, categories: List[str], limit: int = 10) -> List[Dict]:
        """
        Retrieve memories by category membership.
        
        Args:
            categories: List of categories to match
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        results = []
        
        for i, memory in enumerate(self.memories):
            memory_categories = memory.get("categories", [])
            
            # Check if memory belongs to any of the requested categories
            if any(cat in memory_categories for cat in categories):
                results.append({
                    "memory": memory,
                    "memory_id": i,
                    "categories": [cat for cat in categories if cat in memory_categories]
                })
                
        # Sort by recency (newest first)
        results.sort(key=lambda x: x["memory"].get("timestamp", ""), reverse=True)
        
        return results[:limit]
    
    def hybrid_search(self, query: str, categories: List[str] = None, 
                    top_k: int = 5, category_weight: float = 0.3) -> List[Dict]:
        """
        Perform hybrid search combining semantic similarity and category matching.
        This method demonstrates emergent intelligence through combining approaches.
        
        Args:
            query: Search query
            categories: Optional categories to boost
            top_k: Maximum results to return
            category_weight: Weight given to category matches (0.0 to 1.0)
            
        Returns:
            List of memories with combined scores
        """
        # Get base similarity results
        similarity_results = self.retrieve_by_similarity(query, top_k=top_k*2)
        
        # If no categories specified, return similarity results
        if not categories:
            return similarity_results[:top_k]
            
        # Calculate combined scores with category boost
        combined_results = []
        for result in similarity_results:
            memory = result["memory"]
            memory_categories = memory.get("categories", [])
            
            # Base score from similarity
            base_score = result["similarity"]
            
            # Category boost (if memory shares categories with query)
            category_boost = 0.0
            if categories:
                matches = sum(1 for cat in categories if cat in memory_categories)
                if matches > 0:
                    category_boost = matches / max(len(categories), 1) * category_weight
            
            # Combined score
            combined_score = base_score * (1 - category_weight) + category_boost
            
            combined_results.append({
                "memory": memory,
                "memory_id": result["memory_id"],
                "similarity": base_score,
                "category_boost": category_boost,
                "combined_score": combined_score
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return combined_results[:top_k]
    
    def periodic_review(self, llm_client=None):
        """
        Periodically review memory and category organization.
        
        This is a key method for emergent intelligence - the system reflects on and
        improves its own organization based on usage patterns, without complex rules.
        
        Args:
            llm_client: Optional language model client for intelligent suggestions
        """
        if not llm_client or len(self.memories) < 10:
            return
        
        # Sample of recent memories
        recent_memories = self.memories[-20:]
        memory_sample = random.sample(recent_memories, min(10, len(recent_memories)))
        
        # Current categories
        active_categories = {name: info for name, info in self.categories.items() 
                           if info["status"] == "active"}
        
        # Format data for LLM prompt
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
            # Get suggestions from LLM
            response = llm_client.generate(prompt)
            
            # Extract JSON from response
            import re
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group(1))
                
                # Apply suggestions
                self._process_category_suggestions(suggestions)
        except Exception as e:
            print(f"Error in periodic review: {e}")

    def _process_category_suggestions(self, suggestions):
        """Process and apply category organization suggestions."""
        # Add new categories
        for name, description in suggestions.get("new_categories", {}).items():
            if name not in self.categories:
                self.add_category(name, description)
        
        # Update existing categories
        for name, description in suggestions.get("update_categories", {}).items():
            if name in self.categories:
                self.update_category(name, description)
        
        # Process merge suggestions
        for merge in suggestions.get("merge_suggestions", []):
            if len(merge) >= 3:
                cat1, cat2, reason = merge
                if cat1 in self.categories and cat2 in self.categories:
                    # Create relationship to document the merge suggestion
                    self.add_relationship(cat1, "similar_to", cat2, reason)
        
        # Process deprecation suggestions
        for deprecate in suggestions.get("deprecate_categories", []):
            if len(deprecate) >= 2:
                name, reason = deprecate
                if name in self.categories and self.categories[name]["status"] == "active":
                    # Mark as deprecated
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
        
        # Process relationship suggestions
        for rel in suggestions.get("relationships", []):
            if len(rel) >= 4:
                cat1, rel_type, cat2, description = rel
                self.add_relationship(cat1, rel_type, cat2, description)
                
        # Save changes
        self._save_data()
    
    def get_category_evolution(self, category_name):
        """
        Get the evolution history of a specific category.
        
        Args:
            category_name: Name of the category
            
        Returns:
            Dictionary with category history or None if not found
        """
        if category_name not in self.categories:
            return None
            
        category = self.categories[category_name]
        
        # Find related categories through relationships
        related = []
        for rel_id, rel in self.relationships.items():
            if rel["source"] == category_name:
                related.append({
                    "category": rel["target"],
                    "relationship": rel["type"],
                    "description": rel["description"]
                })
            elif rel["target"] == category_name:
                related.append({
                    "category": rel["source"],
                    "relationship": f"inverse of {rel['type']}",
                    "description": rel["description"]
                })
        
        # Find memories using this category
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
            "current": {
                "description": category["description"],
                "status": category["status"],
                "version": category["version"]
            },
            "history": category["history"],
            "related_categories": related,
            "memory_examples": memory_examples
        }
    
    def generate_statistics(self):
        """
        Generate statistics about the memory system usage.
        
        Returns:
            Dictionary of statistics
        """
        # Category usage stats
        category_usage = defaultdict(int)
        for memory in self.memories:
            for category in memory.get("categories", []):
                category_usage[category] += 1
        
        active_categories = sum(1 for c in self.categories.values() if c["status"] == "active")
        deprecated_categories = sum(1 for c in self.categories.values() if c["status"] == "deprecated")
        
        # Evolution stats
        category_versions = {name: info["version"] for name, info in self.categories.items()}
        avg_version = sum(category_versions.values()) / max(1, len(category_versions))
        
        # Relationship stats
        relationship_types = defaultdict(int)
        for rel in self.relationships.values():
            relationship_types[rel["type"]] += 1
        
        return {
            "total_memories": len(self.memories),
            "memories_with_embeddings": sum(1 for m in self.memories if "embedding" in m and m["embedding"]),
            "total_categories": len(self.categories),
            "active_categories": active_categories,
            "deprecated_categories": deprecated_categories,
            "total_relationships": len(self.relationships),
            "category_usage": dict(sorted(category_usage.items(), key=lambda x: x[1], reverse=True)[:10]),
            "average_category_version": avg_version,
            "relationship_types": dict(relationship_types)
        }


# Simple LLM client interface
class SimpleLLMClient:
    """Simple interface for LLM APIs."""
    
    def __init__(self, api_key=None, model=None):
        """
        Initialize LLM client with API key and model.
        
        Args:
            api_key: API key for the provider
            model: Model identifier
        """
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt):
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        raise NotImplementedError("Subclasses must implement this method")


# Example usage
def main():
    """Example usage of the memory system."""
    # Initialize memory system
    memory_system = MemorySystem(storage_path="simplified_memory.json")
    
    # Add some initial memories and categories
    if len(memory_system.memories) == 0:
        memory_system.add_category("technical", "Technical information about systems or code")
        memory_system.add_category("personal", "Personal experiences or opinions")
        
        memory_system.add_memory(
            "Python is a high-level programming language known for its readability.",
            categories=["technical"]
        )
        
        memory_system.add_memory(
            "I really enjoyed learning about neural networks last week.",
            categories=["personal", "technical"]
        )
        
        memory_system.add_relationship("technical", "includes", "programming", 
                                    "Programming is a subcategory of technical topics")
    
    # Interactive loop
    print("\nSimplified Memory System")
    print("=" * 40)
    print(f"Total memories: {len(memory_system.memories)}")
    print(f"Total categories: {len(memory_system.categories)}")
    print("=" * 40)
    
    while True:
        print("\nCommands:")
        print("  add     - Add a new memory")
        print("  search  - Search memories")
        print("  cat     - Manage categories")
        print("  stats   - Show statistics")
        print("  exit    - Exit program")
        
        cmd = input("\nEnter command: ").strip().lower()
        
        if cmd == "exit":
            break
            
        elif cmd == "add":
            content = input("Enter memory content: ")
            cats = input("Categories (comma-separated, leave blank for none): ")
            categories = [c.strip() for c in cats.split(",")] if cats.strip() else []
            
            memory_id = memory_system.add_memory(content, categories=categories)
            print(f"Added memory with ID: {memory_id}")
            
        elif cmd == "search":
            query = input("Enter search query: ")
            cats = input("Filter by categories (comma-separated, leave blank for none): ")
            category_filter = [c.strip() for c in cats.split(",")] if cats.strip() else None
            
            results = memory_system.hybrid_search(query, categories=category_filter)
            
            print("\nSearch Results:")
            for i, result in enumerate(results):
                memory = result["memory"]
                print(f"{i+1}. Score: {result['combined_score']:.3f}")
                print(f"   Content: {memory['content'][:100]}...")
                print(f"   Categories: {memory.get('categories', [])}")
                print()
                
        elif cmd == "cat":
            print("\nCategory Operations:")
            print("  list    - List all categories")
            print("  add     - Add a new category")
            print("  update  - Update a category")
            print("  relate  - Add relationship between categories")
            
            cat_cmd = input("Enter category operation: ").strip().lower()
            
            if cat_cmd == "list":
                print("\nCategories:")
                for name, info in memory_system.categories.items():
                    status = info["status"]
                    print(f"- {name} (v{info['version']}, {status}): {info['description'][:50]}...")
                    
            elif cat_cmd == "add":
                name = input("Enter category name: ")
                desc = input("Enter category description: ")
                if memory_system.add_category(name, desc):
                    print(f"Added category: {name}")
                else:
                    print(f"Category '{name}' already exists")
                    
            elif cat_cmd == "update":
                name = input("Enter category name: ")
                if name in memory_system.categories:
                    desc = input("Enter new description: ")
                    memory_system.update_category(name, desc)
                    print(f"Updated category: {name}")
                else:
                    print(f"Category '{name}' does not exist")
                    
            elif cat_cmd == "relate":
                cat1 = input("First category: ")
                cat2 = input("Second category: ")
                
                if cat1 not in memory_system.categories or cat2 not in memory_system.categories:
                    print("One or both categories don't exist")
                    continue
                    
                rel_type = input("Relationship type (e.g., includes, contradicts): ")
                desc = input("Relationship description: ")
                
                memory_system.add_relationship(cat1, rel_type, cat2, desc)
                print(f"Added relationship: {cat1} {rel_type} {cat2}")
                
        elif cmd == "stats":
            stats = memory_system.generate_statistics()
            
            print("\nMemory System Statistics:")
            print(f"Total memories: {stats['total_memories']}")
            print(f"Memories with embeddings: {stats['memories_with_embeddings']}")
            print(f"Categories: {stats['active_categories']} active, {stats['deprecated_categories']} deprecated")
            print(f"Relationships: {stats['total_relationships']}")
            
            print("\nTop Categories by Usage:")
            for cat, count in list(stats['category_usage'].items())[:5]:
                print(f"- {cat}: {count} memories")
                
            print("\nRelationship Types:")
            for rel_type, count in stats['relationship_types'].items():
                print(f"- {rel_type}: {count}")
                
            print(f"\nAverage category version: {stats['average_category_version']:.1f}")
            
    print("\nExiting memory system")


class OpenAIClient(SimpleLLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        """Initialize with OpenAI API key and model."""
        super().__init__(api_key, model)
        self.client = None
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            print("OpenAI package not installed. Install with: pip install openai")
    
    def generate(self, prompt):
        """Generate text using OpenAI API."""
        if not self.client:
            return "OpenAI client not initialized"
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text: {e}")
            return f"Error: {e}"


if __name__ == "__main__":
    main()