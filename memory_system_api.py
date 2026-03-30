# memory_system_api.py
"""
API for Memory System with Narrative Capabilities

This module provides a FastAPI-based API for the memory system,
enabling web-based interaction and visualization.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import the memory system components
from simple_memory_system import MemorySystem
from enhanced_memory_system import enhance_memory_system
from memory_feedback_loop import MemoryFeedbackLoop
from narrative_memory_system import enhance_with_narrative_capabilities

# Optional LLM client import (uncomment if available)
# from llm_client import DeepSeekClient

# ---- API Data Models ----

class MemoryInput(BaseModel):
    content: str
    categories: List[str] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CategoryInput(BaseModel):
    name: str
    description: str = ""

class RelationshipInput(BaseModel):
    source: str
    relationship_type: str
    target: str
    description: str = ""

class NarrativeInput(BaseModel):
    title: str
    description: str = ""
    memory_ids: List[int] = []
    categories: List[str] = []

class NarrativePointInput(BaseModel):
    narrative_id: str
    content: str
    linked_memories: List[int] = []
    position: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchInput(BaseModel):
    query: str
    categories: List[str] = []
    top_k: int = 5
    use_enhanced: bool = True
    semantic_weight: float = 0.5

class ChatInput(BaseModel):
    message: str
    use_enhanced_search: bool = True
    context_narratives: List[str] = []
    api_key: Optional[str] = None

class MemoryInput(BaseModel):
    content: str
    categories: List[str] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Your other existing models...
class CategoryInput(BaseModel):
    name: str
    description: str = ""



# ---- Initialize Memory System ----

def initialize_memory_system(storage_path="api_memory_store.json"):
    """Initialize the full memory system with all capabilities."""
    # Base memory system
    memory_system = MemorySystem(storage_path=storage_path)
    
    # Enhanced with dual embeddings
    memory_system = enhance_memory_system(memory_system)
    
    # Enhanced with narrative capabilities
    memory_system = enhance_with_narrative_capabilities(memory_system)
    
    # Create feedback loop
    feedback_loop = MemoryFeedbackLoop(memory_system)
    
    return memory_system, feedback_loop

# Initialize the API
app = FastAPI(
    title="Memory System API",
    description="API for Memory System with Narrative Capabilities",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory system
memory_system, feedback_loop = initialize_memory_system()

# ---- API Routes ----

@app.get("/")
async def read_root():
    """API root endpoint."""
    return {"message": "Memory System API is running"}

# ---- Memory Routes ----

@app.post("/memories/", response_model=Dict[str, Any])
async def create_memory(memory: MemoryInput):
    """Add a new memory to the system."""
    memory_id = memory_system.add_memory(
        content=memory.content,
        categories=memory.categories,
        metadata=memory.metadata
    )
    
    # Get the newly created memory
    created_memory = memory_system.memories[memory_id]
    
    # Get category suggestions
    suggestions = []
    if hasattr(memory_system, "suggest_categories_for_memory"):
        suggestions = memory_system.suggest_categories_for_memory(
            memory_id, threshold=0.6
        )
    
    return {
        "memory_id": memory_id,
        "memory": created_memory,
        "category_suggestions": suggestions
    }

@app.get("/memories/", response_model=List[Dict[str, Any]])
async def list_memories(skip: int = 0, limit: int = 100, category: Optional[str] = None):
    """List memories, optionally filtered by category."""
    memories = []
    
    for i, memory in enumerate(memory_system.memories[skip:skip+limit]):
        # Apply category filter if specified
        if category and category not in memory.get("categories", []):
            continue
            
        # Add memory with its ID
        memories.append({
            "memory_id": skip + i,
            "memory": memory
        })
    
    return memories

@app.get("/memories/{memory_id}", response_model=Dict[str, Any])
async def get_memory(memory_id: int):
    """Get a specific memory by ID."""
    if memory_id < 0 or memory_id >= len(memory_system.memories):
        raise HTTPException(status_code=404, detail="Memory not found")
        
    memory = memory_system.memories[memory_id]
    
    return {
        "memory_id": memory_id,
        "memory": memory
    }

@app.post("/memories/{memory_id}/categorize", response_model=Dict[str, Any])
async def categorize_memory(memory_id: int, categories: List[str]):
    """Add categories to a memory."""
    if memory_id < 0 or memory_id >= len(memory_system.memories):
        raise HTTPException(status_code=404, detail="Memory not found")
    
    success = memory_system.categorize_memory(memory_id, categories)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to categorize memory")
    
    return {
        "memory_id": memory_id,
        "categories": memory_system.memories[memory_id].get("categories", []),
        "success": success
    }

# ---- Category Routes ----

@app.post("/categories/", response_model=Dict[str, Any])
async def create_category(category: CategoryInput):
    """Add a new category."""
    success = memory_system.add_category(
        name=category.name,
        description=category.description
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Category already exists")
    
    return {
        "name": category.name,
        "description": category.description,
        "success": success
    }

@app.get("/categories/", response_model=Dict[str, Any])
async def list_categories(status: Optional[str] = None):
    """List all categories, optionally filtered by status."""
    categories = {}
    
    for name, info in memory_system.categories.items():
        # Apply status filter if specified
        if status and info.get("status") != status:
            continue
            
        categories[name] = info
    
    return {
        "categories": categories,
        "count": len(categories)
    }

@app.get("/categories/{name}", response_model=Dict[str, Any])
async def get_category(name: str):
    """Get a specific category by name."""
    if name not in memory_system.categories:
        raise HTTPException(status_code=404, detail="Category not found")
        
    # Get category evolution if available
    evolution = None
    if hasattr(memory_system, "get_category_evolution"):
        evolution = memory_system.get_category_evolution(name)
        
    return {
        "category": memory_system.categories[name],
        "evolution": evolution
    }

@app.put("/categories/{name}", response_model=Dict[str, Any])
async def update_category(name: str, description: str):
    """Update a category description."""
    if name not in memory_system.categories:
        raise HTTPException(status_code=404, detail="Category not found")
        
    success = memory_system.update_category(name, description)
    
    return {
        "name": name,
        "description": description,
        "success": success
    }

# ---- Relationship Routes ----

@app.post("/relationships/", response_model=Dict[str, Any])
async def create_relationship(relationship: RelationshipInput):
    """Add a relationship between categories."""
    success = memory_system.add_relationship(
        category1=relationship.source,
        relationship_type=relationship.relationship_type,
        category2=relationship.target,
        description=relationship.description
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to create relationship")
    
    return {
        "source": relationship.source,
        "type": relationship.relationship_type,
        "target": relationship.target,
        "success": success
    }

@app.get("/relationships/", response_model=Dict[str, Any])
async def list_relationships():
    """List all category relationships."""
    return {
        "relationships": memory_system.relationships,
        "count": len(memory_system.relationships)
    }

# ---- Narrative Routes ----

@app.post("/narratives/", response_model=Dict[str, Any])
async def create_narrative(narrative: NarrativeInput):
    """Create a new narrative."""
    if not hasattr(memory_system, "create_narrative"):
        raise HTTPException(status_code=400, detail="Narrative capabilities not available")
        
    narrative_id = memory_system.create_narrative(
        title=narrative.title,
        description=narrative.description,
        memory_ids=narrative.memory_ids,
        categories=narrative.categories
    )
    
    return {
        "narrative_id": narrative_id,
        "success": bool(narrative_id)
    }

@app.get("/narratives/", response_model=List[Dict[str, Any]])
async def list_narratives(category: Optional[str] = None):
    """List all narratives, optionally filtered by category."""
    if not hasattr(memory_system, "list_narratives"):
        raise HTTPException(status_code=400, detail="Narrative capabilities not available")
        
    narratives = memory_system.list_narratives(category=category)
    
    return narratives

@app.get("/narratives/{narrative_id}", response_model=Dict[str, Any])
async def get_narrative(narrative_id: str):
    """Get a specific narrative by ID."""
    if not hasattr(memory_system, "get_narrative"):
        raise HTTPException(status_code=400, detail="Narrative capabilities not available")
        
    narrative = memory_system.get_narrative(narrative_id)
    
    if not narrative:
        raise HTTPException(status_code=404, detail="Narrative not found")
        
    return narrative

@app.post("/narratives/{narrative_id}/points", response_model=Dict[str, Any])
async def add_narrative_point(point: NarrativePointInput):
    """Add a point to a narrative."""
    if not hasattr(memory_system, "add_narrative_point"):
        raise HTTPException(status_code=400, detail="Narrative capabilities not available")
        
    success = memory_system.add_narrative_point(
        narrative_id=point.narrative_id,
        content=point.content,
        linked_memories=point.linked_memories,
        position=point.position,
        metadata=point.metadata
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add narrative point")
        
    return {
        "narrative_id": point.narrative_id,
        "success": success
    }

# ---- Search Routes ----

@app.post("/search/", response_model=Dict[str, Any])
async def search_memories(search: SearchInput):
    """Search memories with configurable parameters."""
    results = []
    
    if search.use_enhanced and hasattr(memory_system, "enhanced_hybrid_search"):
        # Use enhanced search
        results = memory_system.enhanced_hybrid_search(
            query=search.query,
            categories=search.categories if search.categories else None,
            top_k=search.top_k,
            semantic_weight=search.semantic_weight
        )
    else:
        # Fall back to regular hybrid search
        results = memory_system.hybrid_search(
            query=search.query,
            categories=search.categories if search.categories else None,
            top_k=search.top_k
        )
    
    # Get narrative context if available
    narrative_context = None
    if hasattr(memory_system, "project_narrative_context"):
        narrative_context = memory_system.project_narrative_context(
            query=search.query,
            top_k=3
        )
    
    return {
        "query": search.query,
        "results": results,
        "narrative_context": narrative_context
    }

@app.post("/search/narratives/", response_model=List[Dict[str, Any]])
async def search_narratives(query: str, top_k: int = 5):
    """Search narratives by semantic similarity."""
    if not hasattr(memory_system, "search_narratives"):
        raise HTTPException(status_code=400, detail="Narrative capabilities not available")
        
    results = memory_system.search_narratives(query, top_k=top_k)
    
    return results

# ---- Feedback Routes ----

@app.post("/feedback/run", response_model=Dict[str, Any])
async def run_feedback_cycle(background_tasks: BackgroundTasks, auto_apply: bool = False):
    """Run a feedback cycle to improve the memory system."""
    if not feedback_loop:
        raise HTTPException(status_code=400, detail="Feedback loop not available")
    
    # Run in background to avoid blocking the API
    def run_cycle():
        feedback_loop.run_feedback_cycle(auto_apply=auto_apply)
    
    background_tasks.add_task(run_cycle)
    
    return {
        "message": "Feedback cycle started in background",
        "auto_apply": auto_apply
    }

@app.get("/feedback/suggestions", response_model=Dict[str, Any])
async def get_improvement_suggestions():
    """Get improvement suggestions without applying them."""
    if not feedback_loop:
        raise HTTPException(status_code=400, detail="Feedback loop not available")
        
    suggestions = feedback_loop.suggest_improvements()
    
    return suggestions

@app.get("/feedback/metrics", response_model=Dict[str, Any])
async def get_quality_metrics():
    """Get quality metrics for the memory system."""
    if not feedback_loop:
        raise HTTPException(status_code=400, detail="Feedback loop not available")
        
    metrics = feedback_loop.update_quality_metrics()
    
    return metrics

# ---- Analysis Routes ----

@app.get("/analysis/statistics", response_model=Dict[str, Any])
async def get_system_statistics():
    """Get statistics about the memory system."""
    if not hasattr(memory_system, "generate_statistics"):
        stats = {
            "total_memories": len(memory_system.memories),
            "total_categories": len(memory_system.categories)
        }
    else:
        stats = memory_system.generate_statistics()
    
    return stats

@app.post("/analysis/generate-narrative", response_model=Dict[str, Any])
async def generate_narrative_from_memories(memory_ids: List[int], title: Optional[str] = None):
    """Generate a coherent narrative from a set of memories."""
    if not hasattr(memory_system, "narrative_system") or not hasattr(memory_system.narrative_system, "generate_narrative"):
        raise HTTPException(status_code=400, detail="Narrative generation not available")
    
    # Check if we have an LLM client
    llm_client = None
    # Uncomment if LLM client is available
    # llm_client = DeepSeekClient()
    
    # Generate narrative
    narrative_id = memory_system.narrative_system.generate_narrative(
        memory_ids=memory_ids,
        title=title,
        llm_client=llm_client
    )
    
    if not narrative_id:
        raise HTTPException(status_code=400, detail="Failed to generate narrative")
        
    # Get the generated narrative
    narrative = memory_system.get_narrative(narrative_id)
    
    return {
        "narrative_id": narrative_id,
        "narrative": narrative
    }

@app.post("/analysis/identify-patterns", response_model=List[Dict[str, Any]])
async def identify_narrative_patterns(min_memories: int = 5):
    """Identify patterns across memories that could form coherent narratives."""
    if not hasattr(memory_system, "narrative_system") or not hasattr(memory_system.narrative_system, "identify_narrative_patterns"):
        raise HTTPException(status_code=400, detail="Pattern identification not available")
    
    # Check if we have an LLM client
    llm_client = None
    # Uncomment if LLM client is available
    # llm_client = DeepSeekClient()
    
    # Identify patterns
    patterns = memory_system.narrative_system.identify_narrative_patterns(
        min_memories=min_memories,
        llm_client=llm_client
    )
    
    return patterns

# ---- Helper Routes for Visualization ----

@app.get("/visualization/category-network", response_model=Dict[str, Any])
async def get_category_network():
    """Get category network data for visualization."""
    nodes = []
    links = []
    
    # Add nodes (categories)
    for name, info in memory_system.categories.items():
        # Only include active categories
        if info.get("status") == "deprecated":
            continue
            
        nodes.append({
            "id": name,
            "name": name,
            "description": info.get("description", ""),
            "version": info.get("version", 1),
            # Count memories in this category
            "memory_count": sum(1 for m in memory_system.memories if name in m.get("categories", []))
        })
    
    # Add links (relationships)
    for rel_id, rel in memory_system.relationships.items():
        # Only include relationships between active categories
        source = rel.get("source")
        target = rel.get("target")
        
        if (source in memory_system.categories and target in memory_system.categories and
            memory_system.categories[source].get("status") != "deprecated" and
            memory_system.categories[target].get("status") != "deprecated"):
            
            links.append({
                "source": source,
                "target": target,
                "type": rel.get("type"),
                "description": rel.get("description", "")
            })
    
    return {
        "nodes": nodes,
        "links": links
    }

@app.get("/visualization/memory-timeline", response_model=Dict[str, Any])
async def get_memory_timeline():
    """Get memory timeline data for visualization."""
    timeline_data = []
    
    for i, memory in enumerate(memory_system.memories):
        # Extract timestamp
        timestamp = memory.get("timestamp", "")
        if not timestamp:
            continue
            
        # Create timeline entry
        timeline_data.append({
            "memory_id": i,
            "timestamp": timestamp,
            "content": memory.get("content", "")[:100] + ("..." if len(memory.get("content", "")) > 100 else ""),
            "categories": memory.get("categories", [])
        })
    
    # Sort by timestamp
    timeline_data.sort(key=lambda x: x["timestamp"])
    
    return {
        "timeline": timeline_data
    }

@app.get("/visualization/narrative-graph", response_model=Dict[str, Any])
async def get_narrative_graph():
    """Get narrative graph data for visualization."""
    if not hasattr(memory_system, "narrative_system"):
        raise HTTPException(status_code=400, detail="Narrative capabilities not available")
    
    narratives = []
    
    # Get all narratives
    narrative_list = memory_system.list_narratives()
    
    for narrative_summary in narrative_list:
        narrative_id = narrative_summary.get("id")
        if not narrative_id:
            continue
            
        # Get full narrative
        narrative = memory_system.get_narrative(narrative_id)
        if not narrative:
            continue
            
        # Add to list
        narratives.append(narrative)
    
    # Create graph structure
    nodes = []
    links = []
    
    # Add narrative nodes
    for narrative in narratives:
        nodes.append({
            "id": narrative["id"],
            "type": "narrative",
            "title": narrative["title"],
            "description": narrative.get("description", ""),
            "point_count": len(narrative.get("points", [])),
            "categories": narrative.get("categories", [])
        })
        
        # Add memory nodes and links to narrative
        for point in narrative.get("points", []):
            for memory_id in point.get("linked_memories", []):
                # Add link between narrative and memory
                links.append({
                    "source": narrative["id"],
                    "target": f"memory-{memory_id}",
                    "type": "contains",
                    "point_content": point.get("content", "")[:50] + "..."
                })
                
                # Check if memory node already exists
                memory_node_id = f"memory-{memory_id}"
                if not any(n["id"] == memory_node_id for n in nodes):
                    # Add memory node
                    if memory_id < len(memory_system.memories):
                        memory = memory_system.memories[memory_id]
                        nodes.append({
                            "id": memory_node_id,
                            "type": "memory",
                            "content": memory.get("content", "")[:100] + "...",
                            "categories": memory.get("categories", [])
                        })
    
    # Add links between narratives that share memories
    for i, narrative1 in enumerate(narratives):
        for j, narrative2 in enumerate(narratives[i+1:], i+1):
            # Get memories in each narrative
            memories1 = set()
            memories2 = set()
            
            for point in narrative1.get("points", []):
                memories1.update(point.get("linked_memories", []))
                
            for point in narrative2.get("points", []):
                memories2.update(point.get("linked_memories", []))
            
            # Find shared memories
            shared = memories1.intersection(memories2)
            
            if shared:
                links.append({
                    "source": narrative1["id"],
                    "target": narrative2["id"],
                    "type": "related",
                    "shared_memories": len(shared)
                })
    
    return {
        "nodes": nodes,
        "links": links
    }

# ---- Chat Routes ----

@app.post("/chat/", response_model=Dict[str, Any])
async def chat_with_memory(chat_input: ChatInput):
    """
    Chat endpoint that leverages the memory system for context-aware responses.
    """
    # Search memory for relevant context
    memory_context = {}
    if hasattr(memory_system, "enhanced_hybrid_search") and chat_input.use_enhanced_search:
        search_results = memory_system.enhanced_hybrid_search(
            query=chat_input.message,
            top_k=3
        )
        memory_context["results"] = search_results
    else:
        search_results = memory_system.hybrid_search(
            query=chat_input.message,
            top_k=3
        )
        memory_context["results"] = search_results
    
    # Get narrative context if specified or available
    narrative_context = None
    if hasattr(memory_system, "project_narrative_context"):
        if chat_input.context_narratives:
            narrative_context = memory_system.project_narrative_context(
                query=chat_input.message,
                context_narratives=chat_input.context_narratives
            )
        else:
            narrative_context = memory_system.project_narrative_context(
                query=chat_input.message
            )
        memory_context["narrative_context"] = narrative_context
    
    # Format memory context for the LLM
    context_text = ""
    if memory_context.get("results"):
        context_text = "Relevant information from memory:\n"
        for i, result in enumerate(memory_context["results"]):
            # Format each memory result with complete information
            if isinstance(result, dict) and "memory" in result:
                memory = result["memory"]
                content = memory.get("content", "")
            else:
                memory = result
                content = memory.get("content", "")
                
            context_text += f"{i+1}. User query: {content}\n"
            
            # Include relevant categories if available
            categories = memory.get("categories", [])
            if categories:
                context_text += f"   Categories: {', '.join(categories)}\n"
            
            # Add a separator for readability
            context_text += "\n"
    
    # Add narrative context if available
    if narrative_context and narrative_context.get("has_context"):
        context_text += "\nNarrative Context:\n"
        for i, point in enumerate(narrative_context.get("context_points", [])):
            context_text += f"Narrative point {i+1} ({point['narrative_title']}): {point['content']}\n"
        
        # Add expectations if available
        expectations = narrative_context.get("projection", {}).get("expectations", [])
        if expectations:
            context_text += "\nConsistent patterns in these narratives suggest:\n"
            for exp in expectations:
                context_text += f"- {exp}\n"
    
    # Prepare prompt for LLM
    prompt = f"User question: {chat_input.message}\n\n"
    
    if context_text:
        prompt += f"{context_text}\n"
        prompt += "Based on the above relevant information, narrative context, and the user's question, please provide a helpful response.\n"
    else:
        prompt += "Please provide a helpful response to the user's question.\n"
    
    # Use provided API key or fall back to environment variable
    api_key = chat_input.api_key or os.environ.get("DEEPSEEK_API_KEY")
    
    # Check if DeepSeek client is available
    try:
        # Import DeepSeekClient from chat_with_memory module
        from chat_with_memory import DeepSeekClient
        
        # Initialize with the API key
        deepseek_client = DeepSeekClient(api_key=api_key)
        
        # Check if API key is missing
        if not api_key:
            return {
                "error": "API key is required",
                "message": "Please provide a DeepSeek API key either in the request or as the DEEPSEEK_API_KEY environment variable."
            }
        
        # Generate response
        response = deepseek_client.generate(prompt)
    except ImportError:
        # Fallback response if DeepSeek is not available
        return {
            "error": "LLM integration not available",
            "message": "The DeepSeek client module is not available. Please ensure the necessary dependencies are installed."
        }
    except Exception as e:
        # Handle API errors or other issues
        return {
            "error": "LLM Error",
            "message": f"Error calling DeepSeek API: {str(e)}"
        }
    
    # Store the exchange in memory
    memory_id = memory_system.add_memory(
        content=chat_input.message,
        metadata={
            "type": "conversation",
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "narrative_context": chat_input.context_narratives
        }
    )
    
    # Get and apply category suggestions
    suggestions = []
    if hasattr(memory_system, "suggest_categories_for_memory"):
        suggestions = memory_system.suggest_categories_for_memory(memory_id, threshold=0.65)
        if suggestions:
            suggested_cats = [s["category"] for s in suggestions]
            memory_system.categorize_memory(memory_id, suggested_cats)
    
    # Update narratives if appropriate
    if hasattr(memory_system, "narrative_system") and hasattr(memory_system.narrative_system, "update_narrative_from_conversation"):
        try:
            memory_system.narrative_system.update_narrative_from_conversation(
                user_input=chat_input.message,
                assistant_response=response,
                context_narratives=chat_input.context_narratives,
                llm_client=deepseek_client
            )
        except Exception:
            # Skip narrative updating if there's an issue
            pass
    
    return {
        "message": chat_input.message,
        "response": response,
        "memory_id": memory_id,
        "category_suggestions": suggestions,
        "search_results_count": len(memory_context.get("results", [])),
        "has_narrative_context": bool(narrative_context and narrative_context.get("has_context"))
    }

@app.post("/chat/feedback", response_model=Dict[str, Any])
async def chat_feedback(memory_id: int, helpful: bool, feedback_text: Optional[str] = None):
    """
    Record feedback about a chat interaction to improve future responses.
    """
    if memory_id < 0 or memory_id >= len(memory_system.memories):
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Add feedback to memory metadata
    memory = memory_system.memories[memory_id]
    
    if "metadata" not in memory:
        memory["metadata"] = {}
    
    if "feedback" not in memory["metadata"]:
        memory["metadata"]["feedback"] = []
    
    # Add new feedback
    memory["metadata"]["feedback"].append({
        "helpful": helpful,
        "text": feedback_text,
        "timestamp": datetime.now().isoformat()
    })
    
    # If feedback loop is available, log this feedback
    if feedback_loop and hasattr(feedback_loop, "log_memory_feedback"):
        feedback_loop.log_memory_feedback(
            memory_id=memory_id,
            is_helpful=helpful,
            feedback=feedback_text
        )
    
    return {
        "memory_id": memory_id,
        "success": True
    }

@app.get("/chat/history", response_model=List[Dict[str, Any]])
async def get_chat_history(limit: int = 20):
    """
    Get recent chat history (conversations).
    """
    # Find conversation memories
    conversations = []
    
    # Iterate from newest to oldest
    for i in range(len(memory_system.memories) - 1, -1, -1):
        memory = memory_system.memories[i]
        
        # Check if it's a conversation
        if memory.get("metadata", {}).get("type") == "conversation":
            conversations.append({
                "memory_id": i,
                "message": memory.get("content", ""),
                "response": memory.get("metadata", {}).get("response", ""),
                "timestamp": memory.get("timestamp", ""),
                "categories": memory.get("categories", [])
            })
            
            # Stop after reaching limit
            if len(conversations) >= limit:
                break
    
    return conversations

# Run the API
if __name__ == "__main__":
    uvicorn.run("memory_system_api:app", host="0.0.0.0", port=8000, reload=True)