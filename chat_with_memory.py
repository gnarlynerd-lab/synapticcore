#!/usr/bin/env python3
"""
Chat Interface with Memory Search and Feedback Loop

This script implements an interactive chat interface that integrates:
1. The enhanced memory system with dual embedding spaces
2. The feedback loop for self-improvement
3. Connection to DeepSeek LLM via API
"""

import os
import json
import requests
from datetime import datetime
from simple_memory_system import MemorySystem
from enhanced_memory_system import enhance_memory_system
from memory_feedback_loop import MemoryFeedbackLoop, integrate_with_chat

class DeepSeekClient:
    """Client for interacting with DeepSeek API"""
    
    def __init__(self, api_key=None, api_endpoint="https://api.deepseek.com/v1/chat/completions"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.api_endpoint = api_endpoint

        self.client = None
        
        # Try to initialize OpenAI client if possible
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            print("OpenAI client initialized for DeepSeek API")
        except ImportError:
            print("OpenAI package not installed. Using requests library.")
        
    def generate(self, prompt, system_prompt="You are a helpful assistant."):
        """Generate text using DeepSeek API"""
        # For debugging
        print("\n==== PROMPT SENT TO DEEPSEEK ====")
        print(prompt)
        print("==== END OF PROMPT ====\n")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
        
        try:
            import requests
            response = requests.post(self.api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return f"Error: {e}"

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
    
    # Get the chat function with memory search
    chat_function = integrate_with_chat(memory_system, deepseek_client)
    
    print("\nMemory stats:")
    print(f"  {len(memory_system.memories)} memories loaded")
    print(f"  {len(memory_system.categories)} categories defined")
    
    print("\nSpecial commands:")
    print("  !search <query>  - Search memories without LLM generation")
    print("  !feedback        - Run feedback cycle (analysis only)")
    print("  !feedback apply  - Run feedback cycle and apply changes")
    print("  !metrics         - Show system metrics")
    print("  !categories      - List all categories")
    print("  !exit            - Exit the chat")
    
    # Add search command handler
    def handle_search_command(query):
        """Handle search command without LLM generation"""
        results = feedback_loop.chatbot_memory_search(query)
        
        print("\nSearch Results:")
        if not results["results"]:
            print("  No matching memories found")
            return
            
        for i, result in enumerate(results["results"]):
            print(f"{i+1}. {result['content'][:200]}...")
            print(f"   Categories: {', '.join(result['categories'])}")
            print(f"   Relevance: {result['relevance']:.3f}")
            print()
    
    # Main chat loop
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "!exit":
            break
            
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

if __name__ == "__main__":
    main()