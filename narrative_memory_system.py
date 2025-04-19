#!/usr/bin/env python3
"""
Narrative Memory System

This module extends the existing Memory System with narrative capabilities,
enabling the system to:
1. Form coherent narratives from individual memories
2. Identify causal and temporal relationships between memories
3. Reflect on patterns and themes across memories
4. Project future expectations based on past narratives
"""

import numpy as np
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict

class NarrativePoint:
    """
    Represents a single element in a narrative.
    Can be linked to specific memories or derived from patterns across memories.
    """
    def __init__(self, 
                content: str,
                linked_memories: List[int] = None,
                narrative_id: str = None,
                position: int = None,
                confidence: float = 1.0,
                metadata: Dict = None):
        """
        Initialize a narrative point.
        
        Args:
            content: The narrative statement or observation
            linked_memories: IDs of memories this narrative point is derived from
            narrative_id: ID of the narrative this point belongs to
            position: Relative position in the narrative (temporal or causal order)
            confidence: Confidence level in this narrative element (0.0 to 1.0)
            metadata: Additional metadata about this narrative point
        """
        self.content = content
        self.linked_memories = linked_memories or []
        self.narrative_id = narrative_id
        self.position = position
        self.confidence = confidence
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "linked_memories": self.linked_memories,
            "narrative_id": self.narrative_id,
            "position": self.position,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NarrativePoint':
        """Create from dictionary representation."""
        point = cls(
            content=data["content"],
            linked_memories=data.get("linked_memories", []),
            narrative_id=data.get("narrative_id"),
            position=data.get("position"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )
        point.created_at = data.get("created_at", point.created_at)
        point.updated_at = data.get("updated_at", point.updated_at)
        return point


class Narrative:
    """
    Represents a coherent narrative composed of multiple narrative points.
    """
    def __init__(self,
                title: str,
                description: str = "",
                points: List[NarrativePoint] = None,
                categories: List[str] = None,
                confidence: float = 1.0,
                metadata: Dict = None):
        """
        Initialize a narrative.
        
        Args:
            title: Title of the narrative
            description: Description of the narrative
            points: List of narrative points comprising this narrative
            categories: Categories this narrative belongs to
            confidence: Overall confidence in this narrative (0.0 to 1.0)
            metadata: Additional metadata about this narrative
        """
        self.id = str(uuid.uuid4())
        self.title = title
        self.description = description
        self.points = points or []
        self.categories = categories or []
        self.confidence = confidence
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
    def add_point(self, point: NarrativePoint, position: int = None) -> None:
        """
        Add a narrative point to this narrative.
        
        Args:
            point: Narrative point to add
            position: Position to insert at (if None, appends to end)
        """
        if position is None:
            # Append to end
            position = len(self.points)
            self.points.append(point)
        else:
            # Insert at specified position
            self.points.insert(position, point)
            
            # Update positions of subsequent points
            for i in range(position + 1, len(self.points)):
                if self.points[i].position is not None:
                    self.points[i].position = i
        
        # Update point with narrative information
        point.narrative_id = self.id
        point.position = position
        self.updated_at = datetime.now().isoformat()
        
    def remove_point(self, position: int) -> Optional[NarrativePoint]:
        """
        Remove a narrative point at the specified position.
        
        Args:
            position: Position of the point to remove
            
        Returns:
            Removed narrative point or None if position is invalid
        """
        if position < 0 or position >= len(self.points):
            return None
            
        removed = self.points.pop(position)
        
        # Update positions of subsequent points
        for i in range(position, len(self.points)):
            if self.points[i].position is not None:
                self.points[i].position = i
                
        self.updated_at = datetime.now().isoformat()
        return removed
    
    def reorder_points(self, new_order: List[int]) -> bool:
        """
        Reorder narrative points based on a list of indices.
        
        Args:
            new_order: List of indices specifying the new order
            
        Returns:
            True if successful, False if the new order is invalid
        """
        if len(new_order) != len(self.points) or set(new_order) != set(range(len(self.points))):
            return False
            
        self.points = [self.points[i] for i in new_order]
        
        # Update positions
        for i, point in enumerate(self.points):
            point.position = i
            
        self.updated_at = datetime.now().isoformat()
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "points": [p.to_dict() for p in self.points],
            "categories": self.categories,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Narrative':
        """Create from dictionary representation."""
        narrative = cls(
            title=data["title"],
            description=data.get("description", ""),
            categories=data.get("categories", []),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )
        narrative.id = data.get("id", narrative.id)
        narrative.created_at = data.get("created_at", narrative.created_at)
        narrative.updated_at = data.get("updated_at", narrative.updated_at)
        
        # Add points
        points = [NarrativePoint.from_dict(p) for p in data.get("points", [])]
        for point in points:
            narrative.add_point(point)
            
        return narrative


class NarrativeMemorySystem:
    """
    Extension of the Memory System with narrative capabilities.
    """
    
    def __init__(self, memory_system):
        """
        Initialize the narrative memory system.
        
        Args:
            memory_system: Base memory system to extend
        """
        self.memory_system = memory_system
        self.narratives = {}  # id -> Narrative
        
        # Connect to embedding model if available
        self.embedding_model = getattr(memory_system, "embedding_model", None)
        
        # Initialize narrative point index if embeddings are available
        self.narrative_vector_index = None
        self.narrative_point_to_index = {}  # Maps narrative point ID to vector index
        self.index_to_narrative_point = {}  # Maps vector index to narrative point ID
        
        if self.embedding_model and hasattr(memory_system, "vector_index"):
            self._initialize_narrative_vector_index()
    
    def _initialize_narrative_vector_index(self):
        """Initialize vector index for narrative points."""
        # Need a sample embedding to get dimensions
        if not hasattr(self.memory_system, "vector_index") or not self.memory_system.vector_index:
            return
            
        try:
            import hnswlib
            
            # Get dimension from existing vector index
            dim = self.memory_system.vector_index.dim
            
            # Initialize HNSW index
            self.narrative_vector_index = hnswlib.Index(space='cosine', dim=dim)
            self.narrative_vector_index.init_index(max_elements=1000, ef_construction=200, M=16)
            self.narrative_vector_index.set_ef(50)  # For search
            
            print("Initialized narrative vector index with dimension:", dim)
        except Exception as e:
            print(f"Error initializing narrative vector index: {e}")
    
    def _add_to_narrative_index(self, narrative_id: str, point_idx: int, embedding: List[float]):
        """Add a narrative point embedding to the vector index."""
        if not self.narrative_vector_index:
            return
            
        # Convert to numpy array if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        # Normalize the embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Add to index
        index_id = len(self.narrative_point_to_index)
        self.narrative_vector_index.add_items(embedding, index_id)
        point_id = f"{narrative_id}:{point_idx}"
        self.narrative_point_to_index[point_id] = index_id
        self.index_to_narrative_point[index_id] = point_id
    
    def create_narrative(self, title: str, description: str = "", 
                        memory_ids: List[int] = None,
                        categories: List[str] = None) -> str:
        """
        Create a new narrative, optionally based on specific memories.
        
        Args:
            title: Title of the narrative
            description: Description of the narrative
            memory_ids: IDs of memories to include in this narrative
            categories: Categories for this narrative
            
        Returns:
            ID of the created narrative
        """
        narrative = Narrative(
            title=title,
            description=description,
            categories=categories or []
        )
        
        # Add memories as initial narrative points if provided
        if memory_ids:
            for memory_id in memory_ids:
                if 0 <= memory_id < len(self.memory_system.memories):
                    memory = self.memory_system.memories[memory_id]
                    content = memory.get("content", "")
                    
                    # Create narrative point from memory
                    point = NarrativePoint(
                        content=content,
                        linked_memories=[memory_id],
                        metadata={
                            "source": "direct_memory",
                            "memory_timestamp": memory.get("timestamp", "")
                        }
                    )
                    
                    narrative.add_point(point)
        
        # Store narrative
        self.narratives[narrative.id] = narrative
        
        return narrative.id
    
    def add_narrative_point(self, narrative_id: str, content: str, 
                          linked_memories: List[int] = None,
                          position: int = None,
                          confidence: float = 1.0,
                          metadata: Dict = None) -> bool:
        """
        Add a point to an existing narrative.
        
        Args:
            narrative_id: ID of the narrative
            content: Content of the narrative point
            linked_memories: IDs of memories this point is linked to
            position: Position in the narrative (if None, appends to end)
            confidence: Confidence level (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            True if successful, False if narrative doesn't exist
        """
        if narrative_id not in self.narratives:
            return False
            
        narrative = self.narratives[narrative_id]
        
        # Create narrative point
        point = NarrativePoint(
            content=content,
            linked_memories=linked_memories or [],
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Add to narrative
        narrative.add_point(point, position)
        
        # Create embedding and add to index if embedding model is available
        if self.embedding_model and self.narrative_vector_index:
            try:
                embedding = self.embedding_model.encode(content).tolist()
                point_idx = point.position
                self._add_to_narrative_index(narrative_id, point_idx, embedding)
            except Exception as e:
                print(f"Error creating embedding for narrative point: {e}")
        
        return True
    
    def get_narrative(self, narrative_id: str) -> Optional[Dict]:
        """
        Get a narrative by ID.
        
        Args:
            narrative_id: ID of the narrative
            
        Returns:
            Dictionary representation of the narrative or None if not found
        """
        narrative = self.narratives.get(narrative_id)
        return narrative.to_dict() if narrative else None
    
    def list_narratives(self, category: str = None) -> List[Dict]:
        """
        List all narratives, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of narrative summaries
        """
        summaries = []
        
        for narrative in self.narratives.values():
            # Apply category filter if specified
            if category and category not in narrative.categories:
                continue
                
            # Create summary
            summary = {
                "id": narrative.id,
                "title": narrative.title,
                "description": narrative.description,
                "categories": narrative.categories,
                "point_count": len(narrative.points),
                "created_at": narrative.created_at
            }
            
            summaries.append(summary)
        
        # Sort by creation time (newest first)
        summaries.sort(key=lambda x: x["created_at"], reverse=True)
        
        return summaries
    
    def search_narratives(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search narratives by semantic similarity to query.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            
        Returns:
            List of matching narratives with relevance scores
        """
        if not self.embedding_model or not self.narrative_vector_index:
            # Fallback to simple text matching if embeddings not available
            results = []
            query_lower = query.lower()
            
            for narrative in self.narratives.values():
                # Check title and description for matches
                title_match = query_lower in narrative.title.lower()
                desc_match = query_lower in narrative.description.lower()
                
                # Check narrative points for matches
                points_matching = 0
                for point in narrative.points:
                    if query_lower in point.content.lower():
                        points_matching += 1
                
                if title_match or desc_match or points_matching > 0:
                    # Calculate simple relevance score
                    relevance = 0.0
                    if title_match:
                        relevance += 0.5
                    if desc_match:
                        relevance += 0.3
                    relevance += 0.2 * min(1.0, points_matching / max(1, len(narrative.points)))
                    
                    results.append({
                        "narrative": narrative.to_dict(),
                        "relevance": relevance,
                        "matches": {
                            "title": title_match,
                            "description": desc_match,
                            "points": points_matching
                        }
                    })
            
            # Sort by relevance
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:top_k]
        
        # If embeddings are available, use semantic search
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Normalize embedding
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
                
            # Search index
            labels, distances = self.narrative_vector_index.knn_query(
                query_embedding, 
                k=min(top_k * 3, self.narrative_vector_index.get_current_count())
            )
            
            # Group results by narrative
            narrative_scores = defaultdict(list)
            
            for idx, dist in zip(labels[0], distances[0]):
                similarity = 1.0 - dist
                point_id = self.index_to_narrative_point.get(int(idx))
                
                if point_id:
                    narrative_id, _ = point_id.split(":", 1)
                    narrative_scores[narrative_id].append(similarity)
            
            # Calculate average similarity for each narrative
            results = []
            for narrative_id, similarities in narrative_scores.items():
                if narrative_id in self.narratives:
                    narrative = self.narratives[narrative_id]
                    avg_similarity = sum(similarities) / len(similarities)
                    
                    results.append({
                        "narrative": narrative.to_dict(),
                        "relevance": avg_similarity,
                        "matching_points": len(similarities)
                    })
            
            # Sort by relevance
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"Error in semantic narrative search: {e}")
            return []
    
    def generate_narrative(self, memory_ids: List[int], 
                         title: str = None,
                         llm_client = None) -> Optional[str]:
        """
        Generate a coherent narrative from a set of memories.
        
        Args:
            memory_ids: IDs of memories to include
            title: Optional title for the narrative (auto-generated if None)
            llm_client: Optional language model client for synthesis
            
        Returns:
            ID of the generated narrative or None if generation failed
        """
        if not memory_ids:
            return None
            
        # Filter valid memory IDs
        valid_ids = [mid for mid in memory_ids 
                   if 0 <= mid < len(self.memory_system.memories)]
        
        if not valid_ids:
            return None
            
        # Sort memories by timestamp if available
        memories = []
        for memory_id in valid_ids:
            memory = self.memory_system.memories[memory_id]
            timestamp = memory.get("timestamp", "")
            memories.append((memory_id, memory, timestamp))
            
        # Sort by timestamp
        memories.sort(key=lambda x: x[2])
        
        # Without LLM, create a simple chronological narrative
        if not llm_client:
            if not title:
                # Create a simple title based on categories or time range
                categories = set()
                for _, memory, _ in memories:
                    for cat in memory.get("categories", []):
                        categories.add(cat)
                
                if categories:
                    title = f"Narrative on {', '.join(list(categories)[:3])}"
                else:
                    title = f"Narrative of {len(memories)} memories"
            
            # Create narrative
            narrative_id = self.create_narrative(
                title=title,
                description=f"Automatically generated from {len(memories)} memories",
                memory_ids=[mid for mid, _, _ in memories]
            )
            
            return narrative_id
        
        # With LLM, generate a more coherent narrative
        try:
            # Prepare content for LLM
            memories_text = "\n\n".join([
                f"Memory {i+1} (ID: {mid}, Time: {timestamp}):\n{memory.get('content', '')}"
                for i, (mid, memory, timestamp) in enumerate(memories)
            ])
            
            # Get categories for context
            categories = set()
            for _, memory, _ in memories:
                for cat in memory.get("categories", []):
                    categories.add(cat)
            
            # Create prompt for narrative generation
            prompt = f"""
            Please analyze these related memories and generate a coherent narrative that connects them:
            
            MEMORIES:
            {memories_text}
            
            CATEGORIES: {', '.join(categories) if categories else 'None'}
            
            Generate a narrative with:
            1. A meaningful title that captures the essence of these memories
            2. A brief description of the overall narrative
            3. 3-7 narrative points that form a coherent story, including both explicit content from the memories and reasonable inferences
            4. Suggested categories for this narrative
            
            Format your response as JSON:
            {{
                "title": "Narrative title",
                "description": "Overall description of the narrative",
                "points": [
                    "Point 1 content",
                    "Point 2 content",
                    ...
                ],
                "categories": ["category1", "category2", ...]
            }}
            """
            
            # Get response from LLM
            response = llm_client.generate(prompt)
            
            # Extract JSON from response
            import re
            import json
            json_match = re.search(r'({.*})', response, re.DOTALL)
            
            if json_match:
                narrative_data = json.loads(json_match.group(1))
                
                # Create narrative
                narrative_id = self.create_narrative(
                    title=narrative_data.get("title", title or "Generated Narrative"),
                    description=narrative_data.get("description", f"Generated from {len(memories)} memories"),
                    categories=narrative_data.get("categories", list(categories))
                )
                
                # Add narrative points
                for i, point_content in enumerate(narrative_data.get("points", [])):
                    # Try to link each point to relevant memories
                    linked_memories = []
                    for mid, memory, _ in memories:
                        memory_content = memory.get("content", "").lower()
                        # Simple heuristic: if memory content has significant overlap with point
                        if len(set(memory_content.split()) & set(point_content.lower().split())) > 5:
                            linked_memories.append(mid)
                    
                    self.add_narrative_point(
                        narrative_id=narrative_id,
                        content=point_content,
                        linked_memories=linked_memories,
                        metadata={
                            "source": "llm_generated",
                            "generation_timestamp": datetime.now().isoformat()
                        }
                    )
                
                return narrative_id
            
        except Exception as e:
            print(f"Error generating narrative with LLM: {e}")
        
        # Fallback to simple narrative if LLM fails
        return self.create_narrative(
            title=title or f"Narrative of {len(memories)} memories",
            description=f"Automatically generated from {len(memories)} memories",
            memory_ids=[mid for mid, _, _ in memories]
        )
    
    def reflect_on_memory(self, memory_id: int, llm_client=None) -> List[Dict]:
        """
        Reflect on a memory to find related narratives or create new narrative points.
        
        Args:
            memory_id: ID of the memory to reflect on
            llm_client: Optional language model client for reflection
            
        Returns:
            List of reflection results (narratives affected or created)
        """
        if memory_id < 0 or memory_id >= len(self.memory_system.memories):
            return []
            
        memory = self.memory_system.memories[memory_id]
        memory_content = memory.get("content", "")
        
        # Without LLM, do basic matching to existing narratives
        if not llm_client:
            # Find relevant narratives using semantic search
            if self.embedding_model:
                relevant = self.search_narratives(memory_content, top_k=3)
                
                results = []
                for result in relevant:
                    narrative = result["narrative"]
                    relevance = result["relevance"]
                    
                    # If relevance is high enough, connect memory to narrative
                    if relevance > 0.7:  # Threshold for automatic inclusion
                        result = {
                            "type": "connected_to_narrative",
                            "narrative_id": narrative["id"],
                            "narrative_title": narrative["title"],
                            "relevance": relevance,
                            "action": "added_as_point"
                        }
                        
                        # Add memory as narrative point
                        self.add_narrative_point(
                            narrative_id=narrative["id"],
                            content=memory_content,
                            linked_memories=[memory_id],
                            metadata={
                                "source": "automatic_reflection",
                                "relevance_score": relevance
                            }
                        )
                        
                        results.append(result)
                
                return results
            
            return []
        
        # With LLM, do deeper reflection
        try:
            # Get memory categories and metadata
            categories = memory.get("categories", [])
            timestamp = memory.get("timestamp", "")
            
            # Find potentially related memories
            related_memories = []
            if hasattr(self.memory_system, "retrieve_by_similarity"):
                similar = self.memory_system.retrieve_by_similarity(memory_content, top_k=5)
                related_memories = [
                    (r["memory_id"], r["memory"].get("content", ""), r["similarity"]) 
                    for r in similar if r["memory_id"] != memory_id
                ]
            
            # Find potentially related narratives
            related_narratives = []
            narrative_results = self.search_narratives(memory_content, top_k=3)
            for result in narrative_results:
                narrative = result["narrative"]
                related_narratives.append({
                    "id": narrative["id"],
                    "title": narrative["title"],
                    "relevance": result["relevance"]
                })
            
            # Create prompt for reflection
            related_memories_text = "\n".join([
                f"Related Memory {i+1} (similarity: {sim:.2f}):\n{content}"
                for i, (_, content, sim) in enumerate(related_memories)
            ])
            
            related_narratives_text = "\n".join([
                f"Related Narrative {i+1} (relevance: {n['relevance']:.2f}):\n{n['title']}"
                for i, n in enumerate(related_narratives)
            ])
            
            prompt = f"""
            Please reflect on this memory and its relationship to existing narratives:
            
            MEMORY TO ANALYZE:
            {memory_content}
            
            CATEGORIES: {', '.join(categories) if categories else 'None'}
            TIMESTAMP: {timestamp}
            
            RELATED MEMORIES:
            {related_memories_text if related_memories else "None found"}
            
            RELATED NARRATIVES:
            {related_narratives_text if related_narratives else "None found"}
            
            Please analyze how this memory relates to existing narratives and whether it suggests new narratives.
            Consider:
            1. How does this memory fit into existing narratives?
            2. Does this memory suggest revisions to existing narratives?
            3. Does this memory, combined with related memories, suggest a new narrative?
            
            Format your response as JSON:
            {{
                "existing_narrative_connections": [
                    {{
                        "narrative_id": "id from related narratives or null if new",
                        "connection_type": "extends|contradicts|supports|elaborates",
                        "explanation": "Explanation of how this memory relates to the narrative",
                        "suggested_point": "Suggested narrative point to add (if applicable)"
                    }}
                ],
                "new_narrative_suggestion": {{
                    "suggested": true/false,
                    "title": "Suggested title if new narrative is warranted",
                    "description": "Brief description of the suggested narrative",
                    "initial_points": [
                        "First narrative point",
                        "Second narrative point",
                        ...
                    ],
                    "related_memory_ids": [ids of related memories that should be part of this narrative]
                }}
            }}
            """
            
            # Get response from LLM
            response = llm_client.generate(prompt)
            
            # Extract JSON from response
            import re
            import json
            json_match = re.search(r'({.*})', response, re.DOTALL)
            
            if not json_match:
                return []
                
            reflection_data = json.loads(json_match.group(1))
            results = []
            
            # Process connections to existing narratives
            for connection in reflection_data.get("existing_narrative_connections", []):
                narrative_id = connection.get("narrative_id")
                if not narrative_id or narrative_id not in self.narratives:
                    continue
                    
                connection_type = connection.get("connection_type", "related")
                explanation = connection.get("explanation", "")
                suggested_point = connection.get("suggested_point")
                
                if suggested_point:
                    # Add suggested point to narrative
                    self.add_narrative_point(
                        narrative_id=narrative_id,
                        content=suggested_point,
                        linked_memories=[memory_id],
                        metadata={
                            "source": "llm_reflection",
                            "connection_type": connection_type,
                            "explanation": explanation
                        }
                    )
                    
                    results.append({
                        "type": "connected_to_narrative",
                        "narrative_id": narrative_id,
                        "narrative_title": self.narratives[narrative_id].title,
                        "connection_type": connection_type,
                        "action": "added_as_point"
                    })
            
            # Process suggestion for new narrative
            new_suggestion = reflection_data.get("new_narrative_suggestion", {})
            if new_suggestion.get("suggested", False):
                title = new_suggestion.get("title", "New Narrative")
                description = new_suggestion.get("description", "")
                initial_points = new_suggestion.get("initial_points", [])
                related_ids = new_suggestion.get("related_memory_ids", [])
                
                # Include the current memory
                if memory_id not in related_ids:
                    related_ids.append(memory_id)
                
                # Create new narrative
                narrative_id = self.create_narrative(
                    title=title,
                    description=description,
                    memory_ids=related_ids,
                    categories=memory.get("categories", [])
                )
                
                # Add additional narrative points if provided
                for point_content in initial_points:
                    self.add_narrative_point(
                        narrative_id=narrative_id,
                        content=point_content,
                        linked_memories=[memory_id],
                        metadata={
                            "source": "llm_reflection",
                            "generated_from": "memory_reflection"
                        }
                    )
                
                results.append({
                    "type": "created_narrative",
                    "narrative_id": narrative_id,
                    "narrative_title": title,
                    "related_memory_count": len(related_ids)
                })
            
            return results
            
        except Exception as e:
            print(f"Error in memory reflection: {e}")
            return []
    
    def identify_narrative_patterns(self, min_memories: int = 5, llm_client=None) -> List[Dict]:
        """
        Identify patterns across memories that could form coherent narratives.
        
        Args:
            min_memories: Minimum number of memories needed for pattern detection
            llm_client: Optional language model client for pattern analysis
            
        Returns:
            List of identified patterns and suggested narratives
        """
        if len(self.memory_system.memories) < min_memories:
            return []
            
        # Without LLM, use category co-occurrence for simple pattern detection
        if not llm_client:
            # Find memories by category
            category_memories = defaultdict(list)
            for i, memory in enumerate(self.memory_system.memories):
                for category in memory.get("categories", []):
                    category_memories[category].append(i)
            
            # Find categories with enough memories
            potential_narratives = []
            for category, memory_ids in category_memories.items():
                if len(memory_ids) >= min_memories:
                    potential_narratives.append({
                        "category": category,
                        "memory_count": len(memory_ids),
                        "memory_sample": memory_ids[:5],
                        "narrative_type": "category_based"
                    })
            
            # Sort by memory count
            potential_narratives.sort(key=lambda x: x["memory_count"], reverse=True)
            
            return potential_narratives[:5]  # Return top 5 potential narratives
        
        # With LLM, perform more sophisticated pattern analysis
        try:
            # Sample memories for analysis
            # Use recency-weighted sampling to favor newer memories
            recent_count = min(50, len(self.memory_system.memories))
            recent_memories = [(i, mem) for i, mem in enumerate(self.memory_system.memories[-recent_count:])]
            
            # Prepare memory sample for LLM
            memories_text = "\n\n".join([
                f"Memory {i} (ID: {mem_id}):\nContent: {mem.get('content', '')}\nCategories: {mem.get('categories', [])}"
                for i, (mem_id, mem) in enumerate(recent_memories[:20])  # Limit to 20 memories for prompt size
            ])
            
            # Create prompt for pattern identification
            prompt = f"""
            Please analyze these memories and identify potential narrative patterns:
            
            MEMORY SAMPLE:
            {memories_text}
            
            Please identify 2-4 potential narrative patterns in these memories. Look for:
            1. Recurring themes or topics
            2. Developmental sequences (how ideas or events progress over time)
            3. Causal relationships between memories
            4. Contrasting or opposing viewpoints
            
            For each pattern, suggest a potential narrative that could be created.
            
            Format your response as JSON:
            {{
                "identified_patterns": [
                    {{
                        "pattern_name": "Name of the pattern",
                        "pattern_description": "Description of the identified pattern",
                        "related_memory_indices": [list of indices from the provided memories],
                        "suggested_narrative": {{
                            "title": "Suggested narrative title",
                            "description": "Narrative description",
                            "potential_points": [
                                "First narrative point",
                                "Second narrative point",
                                ...
                            ]
                        }}
                    }}
                ]
            }}
            """
            
            # Get response from LLM
            response = llm_client.generate(prompt)
            
            # Extract JSON from response
            import re
            import json
            json_match = re.search(r'({.*})', response, re.DOTALL)
            
            if not json_match:
                return []
                
            patterns_data = json.loads(json_match.group(1))
            
            # Process identified patterns
            results = []
            for pattern in patterns_data.get("identified_patterns", []):
                pattern_name = pattern.get("pattern_name", "")
                pattern_description = pattern.get("pattern_description", "")
                
                # Map memory indices from the sample to actual memory IDs
                memory_indices = pattern.get("related_memory_indices", [])
                memory_ids = []
                for idx in memory_indices:
                    if 0 <= idx < len(recent_memories):
                        memory_ids.append(recent_memories[idx][0])
                
                # Get suggested narrative
                suggested = pattern.get("suggested_narrative", {})
                
                results.append({
                    "pattern_name": pattern_name,
                    "pattern_description": pattern_description,
                    "memory_count": len(memory_ids),
                    "memory_ids": memory_ids,
                    "suggested_narrative": {
                        "title": suggested.get("title", pattern_name),
                        "description": suggested.get("description", pattern_description),
                        "points": suggested.get("potential_points", [])
                    }
                })
            
            return results
            
        except Exception as e:
            print(f"Error identifying narrative patterns: {e}")
            return []
    
    def project_narrative_context(self, query: str, 
                                context_narratives: List[str] = None,
                                top_k: int = 3,
                                projection_depth: int = 2) -> Dict:
        """
        Project narrative context for a query to influence responses.
        
        This creates a narrative-aware context that can be used to make responses
        more consistent with the system's overall narrative understanding.
        
        Args:
            query: User query to contextualize
            context_narratives: Specific narratives to prioritize (optional)
            top_k: Maximum number of narrative elements to include
            projection_depth: How deeply to analyze connections
            
        Returns:
            Dictionary with narrative context
        """
        # Find relevant narratives for this query
        if context_narratives:
            # Use specified narratives
            relevant_narratives = [
                self.get_narrative(nid) for nid in context_narratives
                if nid in self.narratives
            ]
            relevant_narratives = [n for n in relevant_narratives if n]  # Filter None values
        else:
            # Search for relevant narratives
            search_results = self.search_narratives(query, top_k=top_k)
            relevant_narratives = [result["narrative"] for result in search_results]
        
        if not relevant_narratives:
            return {
                "has_context": False,
                "message": "No relevant narrative context found",
                "projection": None
            }
        
        # Extract key narrative points based on relevance to query
        narrative_points = []
        
        for narrative in relevant_narratives:
            narrative_id = narrative["id"]
            points = narrative.get("points", [])
            
            if self.embedding_model:
                # Use semantic similarity to rank points by relevance to query
                try:
                    query_embedding = self.embedding_model.encode(query)
                    
                    point_similarities = []
                    for i, point in enumerate(points):
                        content = point["content"]
                        point_embedding = self.embedding_model.encode(content)
                        
                        # Calculate cosine similarity
                        similarity = np.dot(query_embedding, point_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(point_embedding)
                        )
                        
                        point_similarities.append((i, point, similarity))
                    
                    # Sort by similarity
                    point_similarities.sort(key=lambda x: x[2], reverse=True)
                    
                    # Take top points from each narrative
                    top_points = point_similarities[:min(3, len(point_similarities))]
                    
                    for _, point, similarity in top_points:
                        if similarity > 0.3:  # Minimum relevance threshold
                            narrative_points.append({
                                "narrative_id": narrative_id,
                                "narrative_title": narrative["title"],
                                "content": point["content"],
                                "relevance": similarity,
                                "confidence": point.get("confidence", 1.0)
                            })
                except Exception as e:
                    print(f"Error in semantic point ranking: {e}")
                    # Fall back to simplified selection
                    for point in points[:2]:  # Take first few points
                        narrative_points.append({
                            "narrative_id": narrative_id,
                            "narrative_title": narrative["title"],
                            "content": point["content"],
                            "relevance": 0.5,  # Default relevance
                            "confidence": point.get("confidence", 1.0)
                        })
            else:
                # Without embeddings, take most recent points
                for point in points[:2]:  # Take first few points
                    narrative_points.append({
                        "narrative_id": narrative_id,
                        "narrative_title": narrative["title"],
                        "content": point["content"],
                        "relevance": 0.5,  # Default relevance
                        "confidence": point.get("confidence", 1.0)
                    })
        
        # Sort narrative points by overall relevance
        narrative_points.sort(key=lambda x: x["relevance"] * x["confidence"], reverse=True)
        
        # Create final context
        context = {
            "has_context": True,
            "relevant_narratives": [
                {
                    "id": n["id"],
                    "title": n["title"],
                    "description": n.get("description", "")
                }
                for n in relevant_narratives[:top_k]
            ],
            "context_points": narrative_points[:top_k],
            "projection": {
                "consistent_themes": self._extract_consistent_themes(narrative_points),
                "expectations": self._generate_expectations(narrative_points, query)
            }
        }
        
        return context
    
    def _extract_consistent_themes(self, narrative_points):
        """Extract consistent themes across narrative points."""
        # Simple implementation - extract common categories
        categories = []
        for point in narrative_points:
            narrative_id = point["narrative_id"]
            if narrative_id in self.narratives:
                categories.extend(self.narratives[narrative_id].categories)
        
        # Count category occurrences
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Return top categories
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in sorted_categories[:3]]
    
    def _generate_expectations(self, narrative_points, query):
        """Generate expectations based on narrative points and query."""
        # Simple implementation - return key patterns
        if not narrative_points:
            return []
            
        # Use common phrases or patterns from narrative points
        expectations = []
        
        # Extract potential expectations from point content
        import re
        for point in narrative_points:
            content = point["content"].lower()
            
            # Look for predictive language
            if re.search(r'(will|should|could|expect|anticipate|predict|likely|future)', content):
                # Extract sentences with predictive language
                sentences = re.split(r'[.!?]', content)
                for sentence in sentences:
                    if re.search(r'(will|should|could|expect|anticipate|predict|likely|future)', sentence):
                        expectations.append(sentence.strip())
        
        # Remove duplicates and limit length
        unique_expectations = []
        for exp in expectations:
            if exp and len(exp) > 10 and exp not in unique_expectations:
                unique_expectations.append(exp)
                if len(unique_expectations) >= 3:
                    break
                    
        return unique_expectations
    
    def update_narrative_from_conversation(self, user_input: str, assistant_response: str, 
                                         context_narratives: List[str] = None,
                                         llm_client=None) -> Dict:
        """
        Update narratives based on conversation between user and assistant.
        
        Args:
            user_input: User's query/input
            assistant_response: Assistant's response
            context_narratives: Narratives used for context in this conversation
            llm_client: Optional language model client for narrative updates
            
        Returns:
            Dictionary with update results
        """
        results = {
            "updated_narratives": [],
            "new_insights": []
        }
        
        # Without LLM, perform simple updates to relevant narratives
        if not llm_client:
            # If we have context narratives, consider adding conversation to them
            if context_narratives:
                for narrative_id in context_narratives:
                    if narrative_id in self.narratives:
                        # Add as new narrative point if conversation adds new information
                        self.add_narrative_point(
                            narrative_id=narrative_id,
                            content=f"User asked: {user_input}\nAssistant responded: {assistant_response}",
                            metadata={
                                "source": "conversation",
                                "user_input": user_input,
                                "assistant_response": assistant_response,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        
                        results["updated_narratives"].append({
                            "narrative_id": narrative_id,
                            "narrative_title": self.narratives[narrative_id].title,
                            "update_type": "added_conversation"
                        })
            
            return results
        
        # With LLM, perform more sophisticated narrative updates
        try:
            # Get current relevant narratives
            relevant_narratives = []
            if context_narratives:
                for narrative_id in context_narratives:
                    if narrative_id in self.narratives:
                        narrative = self.get_narrative(narrative_id)
                        if narrative:
                            relevant_narratives.append(narrative)
            
            if not relevant_narratives:
                # Search for relevant narratives
                search_results = self.search_narratives(user_input, top_k=2)
                relevant_narratives = [result["narrative"] for result in search_results]
            
            # Format narratives for LLM
            narratives_text = ""
            for i, narrative in enumerate(relevant_narratives):
                points_text = "\n".join([
                    f"- {p['content']}" for p in narrative.get("points", [])[:5]
                ])
                
                narratives_text += f"""
                Narrative {i+1}: {narrative['title']}
                Description: {narrative.get('description', '')}
                Points:
                {points_text}
                
                """
            
            # Create prompt for narrative update
            prompt = f"""
            Analyze this conversation and update relevant narratives:
            
            USER: {user_input}
            
            ASSISTANT: {assistant_response}
            
            RELEVANT NARRATIVES:
            {narratives_text if narratives_text else "No existing narratives found."}
            
            Please analyze how this conversation affects our understanding of the relevant narratives.
            Consider:
            1. Does this conversation confirm or contradict existing narrative points?
            2. Does this conversation add new insights to existing narratives?
            3. Should we start a new narrative based on this conversation?
            
            Format your response as JSON:
            {{
                "narrative_updates": [
                    {{
                        "narrative_id": "id from relevant narratives or null if new",
                        "update_type": "confirm|contradict|extend|new_insight",
                        "explanation": "Brief explanation of the update",
                        "new_point": "Text for new narrative point (if applicable)"
                    }}
                ],
                "new_narrative_suggestion": {{
                    "suggested": true/false,
                    "title": "Suggested title if new narrative is warranted",
                    "description": "Brief description of the suggested narrative",
                    "initial_points": [
                        "First narrative point",
                        "Second narrative point"
                    ]
                }}
            }}
            """
            
            # Get response from LLM
            response = llm_client.generate(prompt)
            
            # Extract JSON from response
            import re
            import json
            json_match = re.search(r'({.*})', response, re.DOTALL)
            
            if not json_match:
                return results
                
            update_data = json.loads(json_match.group(1))
            
            # Process narrative updates
            for update in update_data.get("narrative_updates", []):
                narrative_id = update.get("narrative_id")
                if not narrative_id or narrative_id not in self.narratives:
                    continue
                    
                update_type = update.get("update_type", "")
                explanation = update.get("explanation", "")
                new_point = update.get("new_point", "")
                
                if new_point:
                    # Add as new narrative point
                    self.add_narrative_point(
                        narrative_id=narrative_id,
                        content=new_point,
                        metadata={
                            "source": "conversation_update",
                            "update_type": update_type,
                            "explanation": explanation,
                            "user_input": user_input,
                            "assistant_response": assistant_response,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
                    results["updated_narratives"].append({
                        "narrative_id": narrative_id,
                        "narrative_title": self.narratives[narrative_id].title,
                        "update_type": update_type,
                        "explanation": explanation
                    })
            
            # Process new narrative suggestion
            new_suggestion = update_data.get("new_narrative_suggestion", {})
            if new_suggestion.get("suggested", False):
                title = new_suggestion.get("title", "New Conversation Narrative")
                description = new_suggestion.get("description", "")
                initial_points = new_suggestion.get("initial_points", [])
                
                # Create new narrative
                narrative_id = self.create_narrative(
                    title=title,
                    description=description
                )
                
                # Add initial points
                for point_content in initial_points:
                    self.add_narrative_point(
                        narrative_id=narrative_id,
                        content=point_content,
                        metadata={
                            "source": "conversation_derived",
                            "user_input": user_input,
                            "assistant_response": assistant_response,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                
                results["new_insights"].append({
                    "type": "new_narrative",
                    "narrative_id": narrative_id,
                    "narrative_title": title,
                    "point_count": len(initial_points)
                })
            
            return results
            
        except Exception as e:
            print(f"Error updating narrative from conversation: {e}")
            return results


# Integration function for the memory system
def enhance_with_narrative_capabilities(memory_system):
    """
    Enhance an existing memory system with narrative capabilities.
    
    Args:
        memory_system: The base memory system to enhance
        
    Returns:
        Enhanced memory system with narrative capabilities
    """
    # Create narrative memory system
    narrative_system = NarrativeMemorySystem(memory_system)
    
    # Attach to the memory system
    memory_system.narrative_system = narrative_system
    
    # Add helper methods to memory system for convenient access
    memory_system.create_narrative = narrative_system.create_narrative
    memory_system.add_narrative_point = narrative_system.add_narrative_point
    memory_system.get_narrative = narrative_system.get_narrative
    memory_system.list_narratives = narrative_system.list_narratives
    memory_system.search_narratives = narrative_system.search_narratives
    memory_system.project_narrative_context = narrative_system.project_narrative_context
    
    return memory_system


# Example integration with the memory feedback loop
def integrate_narrative_with_feedback(feedback_loop, llm_client=None):
    """
    Integrate narrative capabilities with the memory feedback loop.
    
    Args:
        feedback_loop: The memory feedback loop instance
        llm_client: Optional language model client
        
    Returns:
        Modified feedback loop with narrative integration
    """
    # Get memory system and narrative system
    memory_system = feedback_loop.memory_system
    narrative_system = getattr(memory_system, "narrative_system", None)
    
    if not narrative_system:
        # Enhance memory system with narrative capabilities if not already done
        memory_system = enhance_with_narrative_capabilities(memory_system)
        narrative_system = memory_system.narrative_system
        feedback_loop.memory_system = memory_system
    
    # Store original methods for extension
    original_run_feedback_cycle = feedback_loop.run_feedback_cycle
    
    # Extend the feedback cycle to include narrative reflection
    def extended_run_feedback_cycle(auto_apply=False):
        """Extended feedback cycle with narrative reflection."""
        # Run the original feedback cycle
        results = original_run_feedback_cycle(auto_apply)
        
        # Add narrative reflection
        narrative_results = narrative_system.identify_narrative_patterns(llm_client=llm_client)
        
        # If auto-apply is enabled, create suggested narratives
        if auto_apply and narrative_results:
            for pattern in narrative_results:
                suggested = pattern.get("suggested_narrative", {})
                title = suggested.get("title", pattern.get("pattern_name", "Detected Pattern"))
                description = suggested.get("description", pattern.get("pattern_description", ""))
                points = suggested.get("points", [])
                memory_ids = pattern.get("memory_ids", [])
                
                if memory_ids and (title or description):
                    # Create narrative
                    narrative_id = narrative_system.create_narrative(
                        title=title,
                        description=description,
                        memory_ids=memory_ids[:5]  # Limit to first 5 memories
                    )
                    
                    # Add suggested points
                    for point_content in points:
                        narrative_system.add_narrative_point(
                            narrative_id=narrative_id,
                            content=point_content,
                            metadata={
                                "source": "pattern_detection",
                                "detected_pattern": pattern.get("pattern_name", "")
                            }
                        )
        
        # Add narrative reflection results to overall results
        results["narrative_reflection"] = {
            "patterns_detected": len(narrative_results),
            "patterns": narrative_results
        }
        
        return results
    
    # Replace the feedback cycle method
    feedback_loop.run_feedback_cycle = extended_run_feedback_cycle
    
    # Add narrative-aware search to chatbot
    original_chatbot_memory_search = feedback_loop.chatbot_memory_search
    
    def narrative_aware_memory_search(query, categories=None, result_count=3, use_enhanced_search=False):
        """Enhanced memory search that incorporates narrative context."""
        # Get basic search results
        search_results = original_chatbot_memory_search(query, categories, result_count, use_enhanced_search)
        
        # Get narrative context
        narrative_context = narrative_system.project_narrative_context(query, top_k=2)
        
        # If we have narrative context, include it in results
        if narrative_context and narrative_context.get("has_context"):
            search_results["narrative_context"] = narrative_context
        
        return search_results
    
    # Replace the search method
    feedback_loop.chatbot_memory_search = narrative_aware_memory_search
    
    # Add method to update narratives from conversations
    feedback_loop.update_narratives_from_conversation = narrative_system.update_narrative_from_conversation
    
    return feedback_loop


# Example integration with chat system
def integrate_narrative_with_chat(memory_system, deepseek_client, use_enhanced_search=False):
    """
    Integrate the narrative system with a chat interface.
    
    Args:
        memory_system: Enhanced memory system instance
        deepseek_client: DeepSeek client for LLM integration
        use_enhanced_search: Whether to use enhanced search algorithm
    
    Returns:
        Integrated chat function with narrative awareness
    """
    # Ensure memory system has narrative capabilities
    if not hasattr(memory_system, "narrative_system"):
        memory_system = enhance_with_narrative_capabilities(memory_system)
    
    # Create feedback loop
    from memory_feedback_loop import MemoryFeedbackLoop, integrate_with_chat
    feedback_loop = MemoryFeedbackLoop(memory_system)
    
    # Enhance feedback loop with narrative capabilities
    feedback_loop = integrate_narrative_with_feedback(feedback_loop, deepseek_client)
    
    # Create base chat function
    base_chat_function = integrate_with_chat(memory_system, deepseek_client, use_enhanced_search)
    
    # Define narrative-aware chat function
    def narrative_aware_chat(user_input):
        """Chat function that incorporates narrative context."""
        # Search memory with narrative awareness
        memory_context = feedback_loop.chatbot_memory_search(
            user_input,
            use_enhanced_search=use_enhanced_search
        )
        
        # Extract narrative context if available
        narrative_context = memory_context.get("narrative_context")
        active_narratives = []
        
        context_text = ""
        if memory_context["results"]:
            context_text = "Relevant information from memory:\n"
            for i, result in enumerate(memory_context["results"]):
                context_text += f"{i+1}. User query: {result['content']}\n"
                if result['categories']:
                    context_text += f"   Categories: {', '.join(result['categories'])}\n"
                context_text += "\n"
        
        # Add narrative context to prompt if available
        if narrative_context and narrative_context.get("has_context"):
            context_text += "\nNarrative Context:\n"
            
            # Add narrative points for context
            for i, point in enumerate(narrative_context.get("context_points", [])):
                context_text += f"Narrative point {i+1} ({point['narrative_title']}): {point['content']}\n"
                active_narratives.append(point["narrative_id"])
            
            # Add expectations if available
            expectations = narrative_context.get("projection", {}).get("expectations", [])
            if expectations:
                context_text += "\nConsistent patterns in these narratives suggest:\n"
                for exp in expectations:
                    context_text += f"- {exp}\n"
        
        # Prepare prompt for LLM with narrative awareness
        prompt = f"User question: {user_input}\n\n"
        
        if context_text:
            prompt += f"{context_text}\n"
            prompt += "Based on the above relevant information, narrative context, and the user's question, please provide a helpful response that maintains consistency with established narratives.\n"
        else:
            prompt += "Please provide a helpful response to the user's question.\n"
        
        # Get response from DeepSeek
        response = deepseek_client.generate(prompt)
        
        # Update narratives based on conversation
        feedback_loop.update_narratives_from_conversation(
            user_input=user_input,
            assistant_response=response,
            context_narratives=active_narratives,
            llm_client=deepseek_client
        )
        
        # Store the exchange in memory
        memory_id = memory_system.add_memory(
            content=user_input,
            metadata={
                "type": "conversation",
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "narrative_context": active_narratives
            }
        )
        
        # Run feedback cycle periodically
        if len(memory_system.memories) % 20 == 0:
            feedback_loop.run_feedback_cycle(auto_apply=True)
        
        return response
    
    return narrative_aware_chat