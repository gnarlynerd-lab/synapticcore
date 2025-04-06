#!/usr/bin/env python3
"""
Memory System Feedback Loop with Chatbot Memory Search

This module enables a feedback loop for the Memory-Category Neural Model, allowing:
1. Chatbots to search through the memory system
2. The system to improve categorization based on search patterns
3. Automatic quality evaluation and refinement of categories
"""

import numpy as np
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict

class MemoryFeedbackLoop:
    """
    Implements feedback mechanisms for the Memory-Category Neural Model.
    This enhances the system with self-improvement capabilities.
    """
    
    def __init__(self, memory_system, feedback_log_path="feedback_history.json"):
        """
        Initialize the feedback loop system.
        
        Args:
            memory_system: The enhanced memory system instance
            feedback_log_path: Path to store feedback history
        """
        self.memory_system = memory_system
        self.feedback_log_path = feedback_log_path
        self.feedback_history = self._load_feedback_history()
        
        # Performance tracking
        self.search_patterns = defaultdict(int)  # Track common search terms
        self.category_hits = defaultdict(int)    # Track which categories are useful
        self.failed_searches = []                # Track searches with poor results
        
        # Quality metrics
        self.category_coherence = {}  # How tightly clustered memories are in a category
        self.category_utility = {}    # How useful categories are for retrieval
        
        # Calculate initial metrics
        self.update_quality_metrics()
    
    def _load_feedback_history(self):
        """Load feedback history from storage."""
        if os.path.exists(self.feedback_log_path):
            try:
                with open(self.feedback_log_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading feedback history: {e}")
                return {"feedback_events": [], "metrics_history": []}
        return {"feedback_events": [], "metrics_history": []}
    
    def _save_feedback_history(self):
        """Save feedback history to storage."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.feedback_log_path)), exist_ok=True)
            with open(self.feedback_log_path, 'w') as f:
                json.dump(self.feedback_history, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback history: {e}")
    
    def chatbot_memory_search(self, query: str, categories: List[str] = None, 
                        result_count: int = 3, use_enhanced_search: bool = False) -> Dict:
        """
        Allow a chatbot to search the memory system and get relevant context.
        This function also records search patterns for feedback loop.
        Modified to prevent recursive nesting of conversations.
        
        Args:
            query: The search query
            categories: Optional list of categories to search within
            result_count: Number of results to return
            use_enhanced_search: Whether to use the enhanced search algorithm (if available)
            
        Returns:
            Dictionary with search results and metadata
        """
        # Record search pattern
        search_terms = set(query.lower().split())
        for term in search_terms:
            if len(term) > 3:  # Ignore very short terms
                self.search_patterns[term] += 1
        
        # Perform search - check if enhanced search is available and requested
        start_time = time.time()
        if use_enhanced_search and hasattr(self.memory_system, "enhanced_hybrid_search"):
            # Use enhanced search
            results = self.memory_system.enhanced_hybrid_search(
                query, 
                categories=categories, 
                top_k=result_count
            )
            # Get analysis if available
            analysis = None
            if hasattr(self.memory_system, "analyze_memory_retrieval"):
                try:
                    analysis = self.memory_system.analyze_memory_retrieval(query, results)
                except:
                    pass
        else:
            # Fall back to regular hybrid search
            results = self.memory_system.hybrid_search(
                query, 
                categories=categories, 
                top_k=result_count
            )
            analysis = None
            
        search_time = time.time() - start_time
        
        # Record which categories were useful in results
        found_categories = set()
        for result in results:
            # Handle different result formats
            if isinstance(result, dict) and "memory" in result:
                memory = result["memory"]
            else:
                memory = result
                
            for category in memory.get("categories", []):
                found_categories.add(category)
                self.category_hits[category] += 1
        
        # Record if search was potentially unsuccessful
        avg_score = 0
        if results:
            # Handle different result formats
            if isinstance(results[0], dict) and "combined_score" in results[0]:
                avg_score = sum(r.get("combined_score", 0) for r in results) / len(results)
            elif isinstance(results[0], dict) and "similarity" in results[0]:
                avg_score = sum(r.get("similarity", 0) for r in results) / len(results)
                
            if avg_score < 0.5:
                self.failed_searches.append({
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "avg_score": avg_score
                })
        
        # Format results for chatbot consumption - MODIFIED TO PREVENT RECURSION
        formatted_results = []
        for result in results:
            # Handle different result formats
            if isinstance(result, dict) and "memory" in result:
                memory = result["memory"]
                score = result.get("combined_score", result.get("similarity", 0))
                retrieval_method = result.get("retrieval_method", "search")
            else:
                memory = result
                score = 0
                retrieval_method = "search"
                
            # Extract only the core content, not the full conversation with responses
            content = memory["content"]
            
            # Handle different memory storage formats
            if memory.get("metadata", {}).get("type") == "conversation":
                # Modern format with response in metadata - content is already user message
                pass
            elif "Assistant:" in content:
                # Old format with combined content
                parts = content.split("Assistant:", 1)
                content = parts[0].strip()
                if content.startswith("User:"):
                    content = content[5:].strip()
            
            # Ensure we have substantial content
            if len(content.strip()) < 5:  # Skip very short or empty content
                continue
                
            formatted_results.append({
                "content": content,
                "categories": memory.get("categories", []),
                "relevance": score,
                "timestamp": memory.get("timestamp", ""),
                "retrieval_method": retrieval_method
            })
                
        # Log this search event
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
                
        # Save feedback history
        self._save_feedback_history()
                
        return {
            "results": formatted_results,
            "found_categories": list(found_categories),
            "search_time": search_time,
            "query_terms": list(search_terms),
            "analysis": analysis if analysis else None
        }
    
    def log_category_feedback(self, category_name: str, 
                            is_useful: bool, 
                            memory_id: Optional[int] = None) -> None:
        """
        Log feedback about category usefulness.
        
        Args:
            category_name: The category name
            is_useful: Whether the category was useful
            memory_id: Optional memory ID that this feedback relates to
        """
        feedback = {
            "type": "category_feedback",
            "category": category_name,
            "is_useful": is_useful,
            "memory_id": memory_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback_history["feedback_events"].append(feedback)
        self._save_feedback_history()
    
    def log_memory_categorization(self, memory_id: int, 
                                suggested_categories: List[str],
                                accepted_categories: List[str]) -> None:
        """
        Log feedback about memory categorization suggestions.
        
        Args:
            memory_id: The memory ID
            suggested_categories: Categories suggested by the system
            accepted_categories: Categories actually applied
        """
        feedback = {
            "type": "categorization_feedback",
            "memory_id": memory_id,
            "suggested_categories": suggested_categories,
            "accepted_categories": accepted_categories,
            "acceptance_rate": len(set(suggested_categories) & set(accepted_categories)) / 
                              max(1, len(suggested_categories)),
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback_history["feedback_events"].append(feedback)
        self._save_feedback_history()
    
    def update_quality_metrics(self) -> Dict:
        """
        Update quality metrics for all categories.
        
        Returns:
            Dictionary with quality metrics
        """
        # Skip if no categories or memories
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
        
        # Calculate metrics for each category
        all_coherence_values = []
        categories_with_memories = 0
        
        for category_name in self.memory_system.categories:
            distribution = self.memory_system.get_category_distribution(category_name)
            if not distribution or distribution["member_count"] < 2:
                continue
                
            # Calculate coherence (inverse of variance - higher is more coherent)
            coherence = 0
            if distribution["variance"] and distribution["variance"] > 0:
                coherence = 1.0 / distribution["variance"]
            
            # Utility is based on category hits in searches
            utility = self.category_hits.get(category_name, 0)
            
            # Store metrics
            self.category_coherence[category_name] = coherence
            self.category_utility[category_name] = utility
            
            # Add to metrics record
            metrics["category_metrics"][category_name] = {
                "member_count": distribution["member_count"],
                "coherence": coherence,
                "utility": utility,
                "variance": distribution.get("variance", 0)
            }
            
            categories_with_memories += 1
            if coherence > 0:
                all_coherence_values.append(coherence)
        
        # Calculate system-wide metrics
        categorized_count = 0
        category_count_sum = 0
        
        for memory in self.memory_system.memories:
            categories = memory.get("categories", [])
            if categories:
                categorized_count += 1
                category_count_sum += len(categories)
        
        # Update system metrics
        if self.memory_system.memories:
            metrics["system_metrics"]["categorized_memories_percent"] = (
                categorized_count / len(self.memory_system.memories) * 100
            )
        
        if categorized_count > 0:
            metrics["system_metrics"]["avg_categories_per_memory"] = (
                category_count_sum / categorized_count
            )
        
        if all_coherence_values:
            metrics["system_metrics"]["avg_coherence"] = sum(all_coherence_values) / len(all_coherence_values)
        
        # Save metrics history
        self.feedback_history["metrics_history"].append(metrics)
        self._save_feedback_history()
        
        return metrics
    
    def suggest_improvements(self) -> Dict:
        """
        Suggest improvements to the category system based on feedback.
        
        Returns:
            Dictionary with improvement suggestions
        """
        suggestions = {
            "merge_suggestions": [],
            "split_suggestions": [],
            "new_category_suggestions": [],
            "rename_suggestions": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Find categories that might need to be merged (high similarity, similar utility patterns)
        relationships = self.memory_system.discover_category_relationships(min_similarity=0.75)
        for rel in relationships:
            cat1, cat2 = rel["source"], rel["target"]
            
            # Check if these categories tend to be used together
            cat1_events = [e for e in self.feedback_history["feedback_events"] 
                         if e["type"] == "search" and cat1 in e.get("found_categories", [])]
            cat2_events = [e for e in self.feedback_history["feedback_events"] 
                         if e["type"] == "search" and cat2 in e.get("found_categories", [])]
            
            overlap_events = [e for e in cat1_events if e["query"] in [ce["query"] for ce in cat2_events]]
            
            # If high similarity and often appear together in searches, suggest merge
            if len(overlap_events) >= 3 or (len(cat1_events) > 0 and len(overlap_events) / len(cat1_events) > 0.5):
                suggestions["merge_suggestions"].append({
                    "categories": [cat1, cat2],
                    "similarity": rel["similarity"],
                    "reason": f"High similarity ({rel['similarity']:.2f}) and frequently used together in searches"
                })
        
        # 2. Find categories that might need to be split (low coherence, bimodal distribution)
        for cat_name, coherence in self.category_coherence.items():
            # Low coherence can indicate a category that should be split
            if coherence > 0 and coherence < 0.2:  # Threshold for "poor" coherence
                # Check for outliers that might form a new category
                outliers = self.memory_system.find_category_outliers(cat_name, threshold=1.2)
                if len(outliers) >= 3:  # Need enough outliers to form a new category
                    suggestions["split_suggestions"].append({
                        "category": cat_name,
                        "coherence": coherence,
                        "outlier_count": len(outliers),
                        "reason": f"Low coherence ({coherence:.2f}) with {len(outliers)} outlier memories"
                    })
        
        # 3. Suggest new categories based on search patterns
        common_terms = sorted(self.search_patterns.items(), key=lambda x: x[1], reverse=True)
        
        # Get existing category names for comparison
        existing_categories = set(self.memory_system.categories.keys())
        
        # Check top search terms that aren't already categories
        for term, count in common_terms[:10]:
            if count >= 3 and term not in existing_categories and len(term) > 3:
                # Find memories that might fit this category
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
        
        # 4. Suggest category renames based on low utility
        for cat_name, utility in self.category_utility.items():
            if utility == 0 and cat_name in self.category_coherence:
                # Category exists but is never used in searches
                # Check if there are better search terms that might match
                category_content = []
                for memory in self.memory_system.memories:
                    if cat_name in memory.get("categories", []):
                        category_content.append(memory.get("content", ""))
                
                if category_content:
                    # Look for common words in category content that are also search terms
                    content_words = defaultdict(int)
                    for content in category_content:
                        for word in content.lower().split():
                            if len(word) > 3:
                                content_words[word] += 1
                    
                    # Find overlap with search terms
                    potential_names = []
                    for word, frequency in sorted(content_words.items(), key=lambda x: x[1], reverse=True):
                        if word in self.search_patterns and self.search_patterns[word] >= 2:
                            potential_names.append((word, self.search_patterns[word], frequency))
                    
                    if potential_names:
                        best_name = max(potential_names, key=lambda x: x[1] + x[2])  # Maximize search frequency + content frequency
                        suggestions["rename_suggestions"].append({
                            "current_name": cat_name,
                            "suggested_name": best_name[0],
                            "search_frequency": best_name[1],
                            "content_frequency": best_name[2],
                            "reason": f"Category never used in searches, but '{best_name[0]}' is searched for {best_name[1]} times"
                        })
        
        return suggestions
    
    def apply_improvements(self, improvement_plan: Dict) -> Dict:
        """
        Apply suggested improvements to the memory system.
        
        Args:
            improvement_plan: Dictionary with improvement decisions
            
        Returns:
            Dictionary with results of applied changes
        """
        results = {
            "applied_changes": [],
            "errors": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Apply category merges
        for merge in improvement_plan.get("merges", []):
            try:
                source_cat = merge["source"]
                target_cat = merge["target"]
                
                # Skip if either category doesn't exist
                if source_cat not in self.memory_system.categories or target_cat not in self.memory_system.categories:
                    results["errors"].append(f"Cannot merge: category '{source_cat}' or '{target_cat}' not found")
                    continue
                
                # Get all memories in source category
                source_memories = []
                for i, memory in enumerate(self.memory_system.memories):
                    if source_cat in memory.get("categories", []):
                        source_memories.append(i)
                
                # Add target category to all source memories
                for memory_id in source_memories:
                    self.memory_system.categorize_memory(memory_id, [target_cat])
                
                # Create a relationship to document the merge
                self.memory_system.add_relationship(
                    source_cat, 
                    "merged_into", 
                    target_cat, 
                    f"Categories merged due to similarity and search patterns"
                )
                
                # Mark source category as deprecated if requested
                if merge.get("deprecate_source", False):
                    # Update category status to deprecated
                    category = self.memory_system.categories[source_cat]
                    category["status"] = "deprecated"
                    category["version"] += 1
                    category["updated_at"] = datetime.now().isoformat()
                    category["history"].append({
                        "version": category["version"],
                        "description": category["description"],
                        "status": "deprecated",
                        "reason": f"Merged into {target_cat}",
                        "timestamp": datetime.now().isoformat()
                    })
                
                results["applied_changes"].append({
                    "type": "merge",
                    "source": source_cat,
                    "target": target_cat,
                    "memories_affected": len(source_memories),
                    "deprecated": merge.get("deprecate_source", False)
                })
            except Exception as e:
                results["errors"].append(f"Error merging {merge['source']} into {merge['target']}: {e}")
        
        # 2. Apply category splits
        for split in improvement_plan.get("splits", []):
            try:
                source_cat = split["source"]
                new_cat = split["new_category"]
                
                # Skip if source category doesn't exist
                if source_cat not in self.memory_system.categories:
                    results["errors"].append(f"Cannot split: category '{source_cat}' not found")
                    continue
                
                # Create new category
                self.memory_system.add_category(
                    new_cat, 
                    split.get("description", f"Split from {source_cat}")
                )
                
                # Add relationship
                self.memory_system.add_relationship(
                    new_cat, 
                    "split_from", 
                    source_cat, 
                    f"Category split due to low coherence"
                )
                
                # Categorize memories
                memories_added = 0
                for memory_id in split.get("memories", []):
                    if 0 <= memory_id < len(self.memory_system.memories):
                        self.memory_system.categorize_memory(memory_id, [new_cat])
                        memories_added += 1
                
                results["applied_changes"].append({
                    "type": "split",
                    "source": source_cat,
                    "new_category": new_cat,
                    "memories_added": memories_added
                })
            except Exception as e:
                results["errors"].append(f"Error splitting {split['source']}: {e}")
        
        # 3. Create new categories
        for new_cat in improvement_plan.get("new_categories", []):
            try:
                cat_name = new_cat["name"]
                
                # Skip if category already exists
                if cat_name in self.memory_system.categories:
                    results["errors"].append(f"Cannot create: category '{cat_name}' already exists")
                    continue
                
                # Create new category
                self.memory_system.add_category(
                    cat_name, 
                    new_cat.get("description", f"Automatically created based on search patterns")
                )
                
                # Categorize memories
                memories_added = 0
                for memory_id in new_cat.get("memories", []):
                    if 0 <= memory_id < len(self.memory_system.memories):
                        self.memory_system.categorize_memory(memory_id, [cat_name])
                        memories_added += 1
                
                results["applied_changes"].append({
                    "type": "new_category",
                    "name": cat_name,
                    "memories_added": memories_added
                })
            except Exception as e:
                results["errors"].append(f"Error creating new category {new_cat['name']}: {e}")
        
        # 4. Apply category renames
        for rename in improvement_plan.get("renames", []):
            try:
                old_name = rename["old_name"]
                new_name = rename["new_name"]
                
                # Skip if old category doesn't exist or new one already exists
                if old_name not in self.memory_system.categories:
                    results["errors"].append(f"Cannot rename: category '{old_name}' not found")
                    continue
                    
                if new_name in self.memory_system.categories:
                    results["errors"].append(f"Cannot rename: category '{new_name}' already exists")
                    continue
                
                # Create new category with old description
                old_category = self.memory_system.categories[old_name]
                self.memory_system.add_category(
                    new_name, 
                    rename.get("description", old_category["description"])
                )
                
                # Find all memories with old category
                updated_memories = 0
                for i, memory in enumerate(self.memory_system.memories):
                    if old_name in memory.get("categories", []):
                        # Add new category
                        self.memory_system.categorize_memory(i, [new_name])
                        updated_memories += 1
                
                # Create relationship between categories
                self.memory_system.add_relationship(
                    new_name, 
                    "renamed_from", 
                    old_name, 
                    f"Category renamed for better searchability"
                )
                
                # Mark old category as deprecated if requested
                if rename.get("deprecate_old", False):
                    # Update category status to deprecated
                    old_category["status"] = "deprecated"
                    old_category["version"] += 1
                    old_category["updated_at"] = datetime.now().isoformat()
                    old_category["history"].append({
                        "version": old_category["version"],
                        "description": old_category["description"],
                        "status": "deprecated",
                        "reason": f"Renamed to {new_name}",
                        "timestamp": datetime.now().isoformat()
                    })
                
                results["applied_changes"].append({
                    "type": "rename",
                    "old_name": old_name,
                    "new_name": new_name,
                    "memories_updated": updated_memories,
                    "deprecated_old": rename.get("deprecate_old", False)
                })
            except Exception as e:
                results["errors"].append(f"Error renaming {rename['old_name']} to {rename['new_name']}: {e}")
        
        # Update metrics after changes
        self.update_quality_metrics()
        
        return results
    
    def run_feedback_cycle(self, auto_apply: bool = False) -> Dict:
        """
        Run a complete feedback cycle:
        1. Update quality metrics
        2. Suggest improvements
        3. Optionally auto-apply changes
        
        Args:
            auto_apply: Whether to automatically apply suggested changes
            
        Returns:
            Dictionary with cycle results
        """
        # Update metrics
        metrics = self.update_quality_metrics()
        
        # Get improvement suggestions
        suggestions = self.suggest_improvements()
        
        results = {
            "metrics": metrics,
            "suggestions": suggestions,
            "applied_changes": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Auto-apply changes if requested
        if auto_apply:
            # Convert suggestions to improvement plan
            improvement_plan = {
                "merges": [
                    {
                        "source": merge["categories"][0], 
                        "target": merge["categories"][1],
                        "deprecate_source": False
                    } 
                    for merge in suggestions["merge_suggestions"]
                ],
                "splits": [],  # Need more intelligence to determine which memories go to new category
                "new_categories": [],  # Need more intelligence to determine which memories belong
                "renames": [
                    {
                        "old_name": rename["current_name"],
                        "new_name": rename["suggested_name"],
                        "deprecate_old": False
                    }
                    for rename in suggestions["rename_suggestions"]
                ]
            }
            
            # Apply changes
            applied_results = self.apply_improvements(improvement_plan)
            results["applied_changes"] = applied_results
        
        return results


# Integration with chat system
def integrate_with_chat(memory_system, deepseek_client, use_enhanced_search=False):
    """
    Integrate the feedback loop with a chat system.
    
    Args:
        memory_system: Enhanced memory system instance
        deepseek_client: DeepSeek client for LLM integration
        use_enhanced_search: Whether to use enhanced search algorithm
    
    Returns:
        Integrated chat function
    """
    # Create feedback loop
    feedback_loop = MemoryFeedbackLoop(memory_system)
    
    # Define the chat function
    def chat_with_memory_search(user_input):
        # Skip command processing for feedback loop commands
        if user_input.startswith("!"):
            # [Command processing code remains unchanged]
            return user_input  # If it's not a recognized command, process normally
        
        # Perform memory search to provide context
        memory_context = feedback_loop.chatbot_memory_search(
            user_input,
            use_enhanced_search=use_enhanced_search  # Pass the parameter
        )
        
        # Format memory context for the LLM - IMPROVED
        context_text = ""
        if memory_context["results"]:
            context_text = "Relevant information from memory:\n"
            for i, result in enumerate(memory_context["results"]):
                # Format each memory result with more complete information
                context_text += f"{i+1}. User query: {result['content']}\n"
                
                # Include relevant categories if available
                if result['categories']:
                    context_text += f"   Categories: {', '.join(result['categories'])}\n"
                
                # Add a separator for readability
                context_text += "\n"
        
        # Prepare prompt for LLM - IMPROVED
        prompt = f"User question: {user_input}\n\n"
        
        if context_text:
            prompt += f"{context_text}\n"
            prompt += "Based on the above relevant information and the user's question, please provide a helpful response.\n"
        else:
            prompt += "Please provide a helpful response to the user's question.\n"
        
        # Get response from DeepSeek
        response = deepseek_client.generate(prompt)
        
        # Store the exchange in memory - NEW FORMAT
        memory_id = memory_system.add_memory(
            content=user_input,  # Only store the user's question as content
            metadata={
                "type": "conversation",
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Get and apply category suggestions
        suggestions = memory_system.suggest_categories_for_memory(memory_id, threshold=0.65)
        if suggestions:
            suggested_cats = [s["category"] for s in suggestions]
            memory_system.categorize_memory(memory_id, suggested_cats)
            
            # Log the categorization
            feedback_loop.log_memory_categorization(
                memory_id,
                suggested_cats,
                suggested_cats  # In this case, we accept all suggestions
            )
        
        # Run feedback cycle every 20 exchanges (can adjust this frequency)
        if len(memory_system.memories) % 20 == 0:
            feedback_loop.run_feedback_cycle(auto_apply=True)
        
        return response
    
    return chat_with_memory_search

# Example usage
def main():
    """
    Run the interactive chat interface with memory search and feedback loop.
    """
    print("\n=== Enhanced Memory System Chat with DeepSeek LLM ===")
    
    # Initialize the memory system
    memory_system = MemorySystem(storage_path="memory_store.json")
    memory_system = enhance_memory_system(memory_system)
    
    # Check for API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        api_key = input("Enter your DeepSeek API key (or set DEEPSEEK_API_KEY environment variable): ")
    
    # Initialize DeepSeek client
    deepseek_client = DeepSeekClient(api_key=api_key)
    
    # Initialize feedback loop
    feedback_loop = MemoryFeedbackLoop(memory_system)
    
    # Use enhanced search by default
    use_enhanced_search = True
    
    # Get the chat function with memory search
    chat_function = integrate_with_chat(memory_system, deepseek_client, use_enhanced_search)
    
    print("\nMemory stats:")
    print(f"  {len(memory_system.memories)} memories loaded")
    print(f"  {len(memory_system.categories)} categories defined")
    print(f"  Using {'enhanced' if use_enhanced_search else 'standard'} memory search")
    
    print("\nSpecial commands:")
    print("  !search <query>  - Search memories without LLM generation")
    print("  !feedback        - Run feedback cycle (analysis only)")
    print("  !feedback apply  - Run feedback cycle and apply changes")
    print("  !metrics         - Show system metrics")
    print("  !categories      - List all categories")
    print("  !toggle_search   - Toggle between enhanced and standard search")
    print("  !exit            - Exit the chat")
    
    # Add search command handler
    def handle_search_command(query):
        """Handle search command without LLM generation"""
        results = feedback_loop.chatbot_memory_search(query, use_enhanced_search=use_enhanced_search)
        
        print("\nSearch Results:")
        if not results["results"]:
            print("  No matching memories found")
            return
            
        for i, result in enumerate(results["results"]):
            print(f"{i+1}. {result['content'][:200]}...")
            print(f"   Categories: {', '.join(result['categories'])}")
            print(f"   Relevance: {result['relevance']:.3f}")
            if 'retrieval_method' in result:
                print(f"   Retrieved via: {result['retrieval_method']}")
            print()
        
        # Show analysis if available
        if results.get("analysis"):
            print("\nSearch Analysis:")
            analysis = results["analysis"]
            print(f"  Retrieval methods used: {analysis['retrieval_methods']}")
            if analysis.get("associative_chains"):
                print(f"  Associative chains: {len(analysis['associative_chains'])}")
            print(f"  Category distribution: {analysis['category_distribution']}")
    
    # Main chat loop
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "!exit":
            break
            
        # Handle toggle search command
        if user_input.lower() == "!toggle_search":
            use_enhanced_search = not use_enhanced_search
            print(f"\nSwitched to {'enhanced' if use_enhanced_search else 'standard'} search algorithm")
            chat_function = integrate_with_chat(memory_system, deepseek_client, use_enhanced_search)
            continue
            
        # Handle special search command
        if user_input.lower().startswith("!search "):
            query = user_input[8:].strip()
            handle_search_command(query)
            continue
            
        # Handle categories command
        if user_input.lower() == "!categories":
            print("\nCategories:")
            for name, info in memory_system.categories.items():
                status = info["status"]
                print(f"- {name} ({status}): {info['description'][:80]}...")
            continue

        if user_input.lower().startswith("!addcat "):
            parts = user_input[8:].split(':', 1)
            if len(parts) == 2:
                name, desc = parts[0].strip(), parts[1].strip()
                memory_system.add_category(name, desc)
                print(f"Added category: {name}")
            else:
                print("Usage: !addcat name: description")
            continue
        
        # Process input through the chat function
        response = chat_function(user_input)
        
        if response:  # Some commands return None
            print(f"\nAssistant: {response}")
            
            # Store this exchange
            memory_id = memory_system.add_memory(
                content=user_input,
                    metadata={
                        "type": "conversation",
                        "response": response,
                        "timestamp": datetime.now().isoformat()
                    }
            )   
            
            # Get category suggestions
            suggestions = memory_system.suggest_categories_for_memory(memory_id, threshold=0.5)
            if suggestions:
                suggested_cats = [s["category"] for s in suggestions]
                memory_system.categorize_memory(memory_id, suggested_cats)
    
    print("\nChat session ended. Memory system has been updated and saved.")