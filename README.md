SynapticCore
An open-source cognitive architecture combining advanced memory systems with agent capabilities for intelligent knowledge management.
Overview
SynapticCore is a cognitive architecture that integrates a dual-embedding memory system with LLM-based agent capabilities. Inspired by neuroscience and cognitive psychology theories of memory formation, the system features sophisticated memory retrieval, automatic categorization, and self-improvement mechanisms that enable more effective knowledge management and retrieval.
This project implements both semantic and episodic memory structures with associative retrieval patterns, mimicking how synaptic connections in the brain create networks of related information.
Key Features

Enhanced Memory System: Dual embedding spaces for content and categories
Sophisticated Retrieval: Hybrid search combining semantic, categorical, and associative patterns
Memory Reflection: Feedback loops that improve organization over time
Agent Integration: Proactive knowledge management with planning capabilities
Self-Improvement: Automatic analysis and refinement of memory structures
Web Interface: Interactive UI for knowledge exploration and management

Current Status
This project is under active development. Current components include:

Base memory system with embedding-based retrieval
Enhanced memory system with dual embedding spaces
Feedback mechanisms for memory quality improvement
Integration with LLM APIs (currently supporting DeepSeek)

We are currently implementing the agent framework that will enable more autonomous knowledge management.
Getting Started
Prerequisites

Python 3.8+
Sentence Transformers
HNSWLib for vector indexing
Access to an LLM API (DeepSeek or compatible)

Installation:
# Clone the repository
git clone https://github.com/yourusername/synapticcore.git
cd synapticcore

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export DEEPSEEK_API_KEY=your_api_key_here  # or another compatible LLM API

# Run the interactive system
python chat_with_memory.py

Basic Usage:
from simple_memory_system import MemorySystem
from enhanced_memory_system import enhance_memory_system
from memory_feedback_loop import MemoryFeedbackLoop

# Initialize and enhance memory system
memory_system = MemorySystem()
memory_system = enhance_memory_system(memory_system)

# Add some memories
memory_system.add_memory(
    "Python is a high-level programming language known for its readability.",
    categories=["programming", "technology"]
)

# Search memories
results = memory_system.enhanced_hybrid_search("What programming language is readable?")

Development Roadmap
Phase 1: Core Agent Framework (In Progress)

Agent architecture with reasoning loop
Memory-agent interface
Basic web UI

Phase 2: Agent Capabilities (Planned)

Autonomous knowledge organization
Interactive exploration patterns
Self-improvement mechanisms

Phase 3: Specialization (Planned)

Research assistant capabilities
Enhanced visualization
Evaluation framework

Contributing
Contributions are welcome! Please see our Contributing Guidelines for details on how to get involved.
We're particularly interested in contributions in these areas:

Memory organization algorithms
Agent planning mechanisms
UI/UX improvements
Documentation and examples

Acknowledgments

This project builds on research in cognitive architectures, memory systems, and language models
Special thanks to contributors and early testers

License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
Citation
If you use SynapticCore in your research, please cite:

@software{synapticcore2025,
    author = {Gerard Lynn},
  title = {SynapticCore: A Cognitive Architecture with Dual-Embedding Memory Systems},
  year = {2025},
  url = {https://github.com/yourusername/synapticcore}
  organization = {Integration Information Systems}
}

Copyright 2024 Gerard Lynn

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

