import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from collections import defaultdict

class Memory:
    """Basic memory unit for storing conversation exchanges and metadata."""
    
    def __init__(self, content: str, role: str = "user", timestamp=None):
        self.content = content
        self.role = role  # 'user' or 'assistant'
        self.timestamp = timestamp or datetime.now()
        self.embedding = None  # Will store vector representation
        self.classifications = []  # Self-assigned categories
        self.references = []  # IDs of related memories
        self.metadata = {}  # Additional properties discovered by the agent
        
    def to_dict(self) -> Dict:
        """Convert memory to dictionary for storage."""
        return {
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "classifications": self.classifications,
            "references": self.references,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Memory':
        """Create memory from dictionary."""
        memory = cls(
            content=data["content"],
            role=data["role"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        memory.classifications = data.get("classifications", [])
        memory.references = data.get("references", [])
        memory.metadata = data.get("metadata", {})
        return memory


class SelfClassifyingAgent:
    """
    An agent that maintains conversation history and self-classifies content
    to build its own organizational system for retrieval.
    """
    
    def __init__(self, llm_client=None, embedding_provider=None, storage_path="memory_store.json"):
        """
        Initialize the self-classifying agent.
        
        Args:
            llm_client: Client for language model inference
            embedding_provider: Provider for generating embeddings
            storage_path: Path to store memories
        """
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.storage_path = storage_path
        self.memories = []  # List of Memory objects
        self.memory_index = {}  # Maps memory IDs to indices
        self.classification_system = {
            "categories": {},  # Maps category names to descriptions
            "relations": {},   # Maps relations between categories
            "evolution": []    # Tracks how classification system has evolved
        }
        self.conversation_history = []  # Current conversation
        
        # Load existing data if available
        self._load_data()
        
    def _load_data(self):
        """Load memories and classification system from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # Load memories
                for mem_data in data.get("memories", []):
                    self.memories.append(Memory.from_dict(mem_data))
                
                # Load classification system
                self.classification_system = data.get("classification_system", {
                    "categories": {},
                    "relations": {},
                    "evolution": []
                })
                
                # Rebuild memory index
                self.memory_index = {i: i for i in range(len(self.memories))}
                
                print(f"Loaded {len(self.memories)} memories and {len(self.classification_system['categories'])} categories")
            except Exception as e:
                print(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save memories and classification system to storage."""
        try:
            data = {
                "memories": [mem.to_dict() for mem in self.memories],
                "classification_system": self.classification_system
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved {len(self.memories)} memories and {len(self.classification_system['categories'])} categories")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def add_message(self, content: str, role: str = "user") -> int:
        """
        Add a new message to the conversation history.
        
        Args:
            content: Message content
            role: 'user' or 'assistant'
            
        Returns:
            Index of the new memory
        """
        # Create memory
        memory = Memory(content=content, role=role)
        
        # Add to memories and update index
        memory_idx = len(self.memories)
        self.memories.append(memory)
        self.memory_index[memory_idx] = memory_idx
        
        # Add to current conversation history
        self.conversation_history.append((role, content))
        
        # If this is an assistant response, classify it and update references
        if role == "assistant" and len(self.conversation_history) > 1:
            self._classify_memory(memory_idx)
            
            # Link to the previous user message
            if len(self.memories) >= 2:
                memory.references.append(memory_idx - 1)
        
        # If this completes an exchange, trigger classification for the user message too
        if role == "assistant" and len(self.memories) >= 2:
            prev_idx = memory_idx - 1
            if self.memories[prev_idx].role == "user" and not self.memories[prev_idx].classifications:
                self._classify_memory(prev_idx)
        
        # Periodically review and update classification system
        if len(self.memories) % 10 == 0:
            self._update_classification_system()
        
        # Save data
        self._save_data()
        
        return memory_idx
    
    def _classify_memory(self, memory_idx: int):
        """
        Self-classify a memory based on content and context.
        
        Args:
            memory_idx: Index of memory to classify
        """
        if not self.llm_client:
            return
        
        memory = self.memories[memory_idx]
        
        # Get existing classification system
        categories = list(self.classification_system["categories"].items())
        categories_text = "\n".join([f"- {name}: {desc}" for name, desc in categories])
        
        # If we have no categories yet, start with an empty system
        if not categories:
            categories_text = "(No existing categories yet)"
        
        # Get context from surrounding messages
        context = self._get_context_for_memory(memory_idx)
        
        # Create classification prompt
        prompt = f"""
        I need to categorize and enrich the following message in a conversation:

        [Message]: {memory.content}
        [Role]: {memory.role}
        
        Context:
        {context}
        
        Current classification categories:
        {categories_text}
        
        Please perform three tasks:
        
        1. Assign appropriate categories to this message. You can use existing categories or create new ones if needed.
        For each new category, provide a brief description of what it encompasses.
        
        2. Identify key concepts, entities, or themes in this message that should be recorded as metadata.
        
        3. If this message relates to or references previous conversations or topics, note these relationships.
        
        Format your response as JSON with these fields:
        {{
            "categories": ["category1", "category2"],
            "new_categories": {{"category_name": "category description"}},
            "metadata": {{"key1": "value1", "key2": "value2"}},
            "references": ["concept/topic this relates to"]
        }}
        """
        
        try:
            response = self.llm_client.generate(prompt)
            
            # Extract JSON from response
            import re
            import json
            
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                classification_data = json.loads(json_match.group(0))
                
                # Update memory with classifications
                memory.classifications = classification_data.get("categories", [])
                
                # Add metadata
                memory.metadata.update(classification_data.get("metadata", {}))
                
                # Add new categories to classification system
                for cat_name, cat_desc in classification_data.get("new_categories", {}).items():
                    if cat_name not in self.classification_system["categories"]:
                        self.classification_system["categories"][cat_name] = cat_desc
                
                # Handle references (conceptual, not just memory IDs)
                memory.metadata["conceptual_references"] = classification_data.get("references", [])
        
        except Exception as e:
            print(f"Error classifying memory: {e}")
    
    def _get_context_for_memory(self, memory_idx: int) -> str:
        """Get surrounding context for a memory."""
        memory = self.memories[memory_idx]
        
        # Get immediate conversation context (3 messages before)
        start_idx = max(0, memory_idx - 3)
        context_memories = self.memories[start_idx:memory_idx]
        
        context = "Recent conversation:\n"
        for mem in context_memories:
            context += f"[{mem.role}]: {mem.content}\n"
        
        return context
    
    def _update_classification_system(self):
        """Periodically review and refine the classification system."""
        if not self.llm_client or len(self.memories) < 5:
            return
        
        # Get current categories and usage statistics
        categories = self.classification_system["categories"]
        category_usage = defaultdict(int)
        
        for memory in self.memories:
            for category in memory.classifications:
                category_usage[category] += 1
        
        # Format for the prompt
        categories_text = "\n".join([
            f"- {name} ({category_usage.get(name, 0)} uses): {desc}" 
            for name, desc in categories.items()
        ])
        
        # Get sample of recent memories
        recent_memories = self.memories[-20:]
        samples_text = "\n".join([
            f"[{mem.role}]: {mem.content}\n  Categories: {mem.classifications}"
            for mem in recent_memories[-5:]
        ])
        
        prompt = f"""
        Please analyze my current conversation classification system and suggest improvements.
        
        Current categories and usage:
        {categories_text}
        
        Sample of recent messages and their classifications:
        {samples_text}
        
        Based on this analysis:
        
        1. Suggest any new categories that would be useful
        2. Suggest any categories that should be merged or split
        3. Identify relationships between categories (hierarchical or associative)
        
        Format your response as JSON:
        {{
            "new_categories": {{"category_name": "category description"}},
            "merge_suggestions": [["category1", "category2", "merged_name", "merged_description"]],
            "split_suggestions": [["category_to_split", [["new_cat1", "desc1"], ["new_cat2", "desc2"]]]],
            "relationships": [["category1", "relationship_type", "category2", "description"]]
        }}
        """
        
        try:
            response = self.llm_client.generate(prompt)
            
            # Extract JSON from response
            import re
            import json
            
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                update_data = json.loads(json_match.group(0))
                
                # Track this evolution
                self.classification_system["evolution"].append({
                    "timestamp": datetime.now().isoformat(),
                    "changes": update_data,
                    "categories_before": len(categories)
                })
                
                # Add new categories
                for cat_name, cat_desc in update_data.get("new_categories", {}).items():
                    if cat_name not in categories:
                        categories[cat_name] = cat_desc
                
                # Handle merges
                for merge in update_data.get("merge_suggestions", []):
                    if len(merge) == 4:
                        cat1, cat2, merged_name, merged_desc = merge
                        if cat1 in categories and cat2 in categories:
                            # Create merged category
                            categories[merged_name] = merged_desc
                            
                            # Update all memories with the merged category
                            for memory in self.memories:
                                if cat1 in memory.classifications or cat2 in memory.classifications:
                                    memory.classifications = [c for c in memory.classifications 
                                                           if c != cat1 and c != cat2]
                                    memory.classifications.append(merged_name)
                            
                            # Remove old categories
                            if cat1 in categories:
                                del categories[cat1]
                            if cat2 in categories:
                                del categories[cat2]
                
                # Handle relationships
                for rel in update_data.get("relationships", []):
                    if len(rel) == 4:
                        cat1, rel_type, cat2, desc = rel
                        rel_key = f"{cat1}_{rel_type}_{cat2}"
                        self.classification_system["relations"][rel_key] = {
                            "from": cat1,
                            "to": cat2,
                            "type": rel_type,
                            "description": desc
                        }
                
                print(f"Updated classification system with {len(update_data.get('new_categories', {}))} new categories")
        
        except Exception as e:
            print(f"Error updating classification system: {e}")
    
    def generate_response(self, user_message: str) -> str:
        """
        Generate a response to a user message, incorporating relevant past context.
        
        Args:
            user_message: User's message
            
        Returns:
            Agent's response
        """
        if not self.llm_client:
            return "Sorry, I don't have a language model configured."
        
        # Add user message to memory
        user_memory_idx = self.add_message(user_message, "user")
        
        # Classify the new message immediately to use for retrieval
        self._classify_memory(user_memory_idx)
        
        # Retrieve relevant memories
        relevant_memories = self._retrieve_relevant_memories(user_memory_idx)
        
        # Format conversation history and relevant context
        conversation_context = "\n".join([
            f"{'User' if role == 'user' else 'Assistant'}: {message}"
            for role, message in self.conversation_history[-5:]
        ])
        
        relevant_context = ""
        if relevant_memories:
            relevant_context = "Relevant information from past conversations:\n"
            for mem_idx, relevance in relevant_memories:
                memory = self.memories[mem_idx]
                relevant_context += f"- {memory.content}\n"
        
        # Generate response
        prompt = f"""
        Conversation history:
        {conversation_context}
        
        {relevant_context}
        
        Generate a helpful response to the user's last message.
        """
        
        try:
            response = self.llm_client.generate(prompt)
            
            # Add response to memory
            self.add_message(response, "assistant")
            
            return response
        except Exception as e:
            error_response = f"I'm having trouble generating a response: {e}"
            self.add_message(error_response, "assistant")
            return error_response
    
    def _retrieve_relevant_memories(self, memory_idx: int) -> List[Tuple[int, float]]:
        """
        Retrieve memories relevant to the given memory.
        
        Args:
            memory_idx: Index of the memory to find relevant items for
            
        Returns:
            List of (memory_idx, relevance_score) tuples
        """
        memory = self.memories[memory_idx]
        
        # If we have too few memories, return an empty list
        if len(self.memories) < 5:
            return []
        
        # Start with classification-based retrieval
        # Find memories with matching classifications
        matching_category_indices = []
        for idx, mem in enumerate(self.memories):
            # Skip current memory and recent conversation
            if idx == memory_idx or idx > memory_idx - 5:
                continue
                
            # Check for category matches
            shared_categories = set(mem.classifications) & set(memory.classifications)
            if shared_categories:
                matching_category_indices.append((idx, len(shared_categories) / 
                                                len(set(mem.classifications) | 
                                                   set(memory.classifications))))
        
        # If we have embedding_provider, use semantic search as well
        semantic_matches = []
        if self.embedding_provider and memory.embedding is not None:
            # Ensure all memories have embeddings
            for idx, mem in enumerate(self.memories):
                if mem.embedding is None and isinstance(mem.content, str):
                    try:
                        mem.embedding = self.embedding_provider.get_embedding(mem.content)
                    except:
                        pass
            
            # Find semantically similar memories
            for idx, mem in enumerate(self.memories):
                if idx == memory_idx or idx > memory_idx - 5 or mem.embedding is None:
                    continue
                
                similarity = self._vector_similarity(memory.embedding, mem.embedding)
                if similarity > 0.7:  # Threshold for similarity
                    semantic_matches.append((idx, similarity))
        
        # Combine results, prioritizing both category and semantic matches
        combined_results = {}
        
        # Add category matches
        for idx, score in matching_category_indices:
            combined_results[idx] = score
        
        # Add semantic matches, boosting scores that appear in both
        for idx, score in semantic_matches:
            if idx in combined_results:
                combined_results[idx] = (combined_results[idx] + score) / 2 + 0.1  # Boost for dual match
            else:
                combined_results[idx] = score * 0.8  # Slightly lower weight for semantic-only matches
        
        # If we have too few matches, use LLM to find relevant memories
        if len(combined_results) < 2 and self.llm_client:
            context_matches = self._retrieve_by_llm(memory)
            for idx, score in context_matches:
                if idx not in combined_results:
                    combined_results[idx] = score
        
        # Sort by relevance score and take top 5
        results = [(idx, score) for idx, score in combined_results.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:5]
    
    def _vector_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
            
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot / (norm1 * norm2)
    
    def _retrieve_by_llm(self, memory: Memory) -> List[Tuple[int, float]]:
        """Use LLM to determine relevant past memories."""
        if not self.llm_client:
            return []
        
        # Get a sample of past memories (excluding very recent ones)
        if len(self.memories) <= 10:
            return []
            
        # Sample some older memories
        sample_size = min(20, len(self.memories) - 5)
        memory_indices = list(range(len(self.memories) - 5))
        
        if sample_size < len(memory_indices):
            import random
            memory_indices = random.sample(memory_indices, sample_size)
        
        # Format memories for the prompt
        memories_text = "\n".join([
            f"{i}: {self.memories[idx].content}"
            for i, idx in enumerate(memory_indices)
        ])
        
        prompt = f"""
        I need to find past conversation messages that are relevant to this new message:
        
        [New message]: {memory.content}
        
        Here are some past messages:
        {memories_text}
        
        List the numbers of the 3 most relevant past messages that would provide helpful context.
        Rank them by relevance score from 0.0 to 1.0.
        
        Format your response as JSON:
        {{
            "relevant_indices": [
                [message_number, relevance_score],
                [message_number, relevance_score],
                [message_number, relevance_score]
            ]
        }}
        """
        
        try:
            response = self.llm_client.generate(prompt)
            
            # Extract JSON from response
            import re
            import json
            
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                # Map indices from the prompt to actual memory indices
                results = []
                for idx, score in data.get("relevant_indices", []):
                    if 0 <= idx < len(memory_indices):
                        results.append((memory_indices[idx], float(score)))
                
                return results
                
        except Exception as e:
            print(f"Error retrieving by LLM: {e}")
            
        return []
    
    def get_classification_stats(self) -> Dict:
        """Get statistics about the classification system."""
        if not self.memories:
            return {"error": "No memories available"}
        
        # Count category usage
        category_counts = defaultdict(int)
        for memory in self.memories:
            for category in memory.classifications:
                category_counts[category] += 1
        
        # Get top categories
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Count memories by role
        role_counts = defaultdict(int)
        for memory in self.memories:
            role_counts[memory.role] += 1
        
        # Get evolution metrics
        evolution_steps = len(self.classification_system["evolution"])
        current_categories = len(self.classification_system["categories"])
        
        return {
            "total_memories": len(self.memories),
            "roles": dict(role_counts),
            "top_categories": top_categories[:10],
            "total_categories": current_categories,
            "evolution_steps": evolution_steps,
            "category_relations": len(self.classification_system["relations"])
        }
        
    def visualize_classification_system(self, output_file="classification_system.html"):
        """Create a visualization of the classification system."""
        if not self.classification_system["categories"]:
            return {"error": "No categories available"}
            
        try:
            # Create a simple HTML visualization
            categories = self.classification_system["categories"]
            relations = self.classification_system["relations"]
            
            # Count category usage
            category_counts = defaultdict(int)
            for memory in self.memories:
                for category in memory.classifications:
                    category_counts[category] += 1
            
            # Create HTML
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Self-Organizing Classification System</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .category { 
                        border: 1px solid #ccc; 
                        padding: 10px; 
                        margin: 10px; 
                        border-radius: 5px;
                        display: inline-block;
                        min-width: 200px;
                    }
                    .relation {
                        margin: 20px;
                        padding: 10px;
                        background-color: #f0f0f0;
                        border-radius: 5px;
                    }
                    .count { font-weight: bold; }
                    .stats { margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <h1>Self-Organizing Classification System</h1>
                
                <div class="stats">
                    <p>Total Categories: """ + str(len(categories)) + """</p>
                    <p>Total Relations: """ + str(len(relations)) + """</p>
                    <p>Total Memories: """ + str(len(self.memories)) + """</p>
                </div>
                
                <h2>Categories</h2>
                <div class="categories">
            """
            
            # Add categories
            for cat_name, cat_desc in categories.items():
                count = category_counts.get(cat_name, 0)
                html += f"""
                <div class="category">
                    <h3>{cat_name}</h3>
                    <p>{cat_desc}</p>
                    <p class="count">Used: {count} times</p>
                </div>
                """
            
            # Add relations
            html += """
                </div>
                
                <h2>Relations</h2>
                <div class="relations">
            """
            
            for rel_key, rel_data in relations.items():
                html += f"""
                <div class="relation">
                    <p><strong>{rel_data['from']}</strong> {rel_data['type']} <strong>{rel_data['to']}</strong></p>
                    <p>{rel_data['description']}</p>
                </div>
                """
            
            # Close HTML
            html += """
                </div>
            </body>
            </html>
            """
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(html)
            
            return {"file": output_file}
            
        except Exception as e:
            return {"error": f"Visualization error: {e}"}


# DeepSeek LLM Client
class DeepSeekClient:
    """Simple DeepSeek API client for the self-classifying agent."""
    
    def __init__(self, api_key, model="deepseek-chat", base_url="https://api.deepseek.com"):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key
            model: Model to use (default: "deepseek-chat")
            base_url: API base URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        try:
            # DeepSeek uses OpenAI compatible interface
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        except ImportError:
            print("OpenAI Python package not found. Install it with: pip install openai")
            self.client = None
    
    def generate(self, prompt: str) -> str:
        """Generate text using DeepSeek API."""
        if not self.client:
            return "DeepSeek client not initialized"
        
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
            print(f"DeepSeek API error: {e}")
            return f"Error: {e}"


# DeepSeek embedding provider
class DeepSeekEmbedding:
    """Provides embeddings using DeepSeek API."""
    
    def __init__(self, api_key, model="deepseek-embedding", base_url="https://api.deepseek.com"):
        """
        Initialize embedding provider with DeepSeek.
        
        Args:
            api_key: DeepSeek API key
            model: Embedding model to use
            base_url: API base URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        try:
            # DeepSeek uses OpenAI compatible interface
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        except ImportError:
            print("OpenAI Python package not found. Install it with: pip install openai")
            self.client = None
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using DeepSeek API."""
        if not self.client:
            return None
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            return response.data[0].embedding
        except Exception as e:
            print(f"DeepSeek embedding error: {e}")
            return None
            
    
# Alternative embedding provider using SentenceTransformers (local, no API needed)
class LocalEmbedding:
    """Provides embeddings using local SentenceTransformers models."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize local embedding provider.
        
        Args:
            model_name: Name of the SentenceTransformers model to use
        """
        self.model_name = model_name
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"Loaded local embedding model: {model_name}")
        except ImportError:
            print("SentenceTransformers not found. Install with: pip install sentence-transformers")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using local model."""
        if self.model is None:
            return None
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except Exception as e:
            print(f"Local embedding error: {e}")
            return None


# Example usage
def main():
    import os
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run Self-Classifying Agent")
    parser.add_argument("--model", type=str, default="deepseek-chat", 
                      help="Model to use (default: deepseek-chat)")
    parser.add_argument("--embedding-type", type=str, choices=["deepseek", "local"], default="local",
                      help="Type of embedding to use (default: local)")
    parser.add_argument("--storage", type=str, default="memory_store.json",
                      help="Storage file path (default: memory_store.json)")
    args = parser.parse_args()
    
    # Get DeepSeek API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        api_key = input("Enter your DeepSeek API key: ")
        os.environ["DEEPSEEK_API_KEY"] = api_key
    
    # Initialize LLM client
    llm_client = DeepSeekClient(api_key, model=args.model)
    
    # Initialize embedding provider
    if args.embedding_type == "deepseek":
        embedding_provider = DeepSeekEmbedding(api_key)
        print("Using DeepSeek for embeddings")
    else:
        embedding_provider = LocalEmbedding()
        print("Using local SentenceTransformers for embeddings")
    
    # Initialize agent
    agent = SelfClassifyingAgent(
        llm_client=llm_client,
        embedding_provider=embedding_provider,
        storage_path=args.storage
    )
    
    print("Self-Classifying Agent initialized. Type 'exit' to end conversation.")
    print("Type 'stats' to see classification statistics.")
    print("Type 'viz' to generate visualization.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'stats':
            stats = agent.get_classification_stats()
            print("\nClassification Stats:")
            for key, value in stats.items():
                print(f"{key}: {value}")
        elif user_input.lower() == 'viz':
            result = agent.visualize_classification_system()
            print(f"\nVisualization: {result}")
        else:
            response = agent.generate_response(user_input)
            print(f"\nAgent: {response}")


if __name__ == "__main__":
    main()