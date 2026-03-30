"""
Tests for the refactored synapticcore package.

Mirrors test_core.py but imports from the new package structure.
"""

import os
import pytest
import numpy as np

from synapticcore import SynapticCore, MemorySystem
from synapticcore.storage import JsonFileStore


# ─── MemorySystem Tests ────────────────────────────────────────────────────

class TestMemorySystem:

    def test_add_memory_returns_id(self, memory_system):
        assert memory_system.add_memory("test content") == 0

    def test_add_memory_sequential_ids(self, memory_system):
        assert memory_system.add_memory("first") == 0
        assert memory_system.add_memory("second") == 1

    def test_memory_has_embedding(self, memory_system):
        memory_system.add_memory("test content with enough words for embedding")
        assert memory_system.memories[0]["embedding"] is not None

    def test_memory_has_categories(self, memory_system):
        memory_system.add_category("tech", "technology")
        memory_system.add_memory("test", categories=["tech"])
        assert "tech" in memory_system.memories[0]["categories"]

    def test_add_memory_auto_creates_category(self, memory_system):
        memory_system.add_memory("test", categories=["new_cat"])
        assert "new_cat" in memory_system.categories

    def test_add_category(self, memory_system):
        assert memory_system.add_category("tech", "Technology") is True
        assert memory_system.categories["tech"]["version"] == 1

    def test_add_duplicate_category(self, memory_system):
        memory_system.add_category("tech")
        assert memory_system.add_category("tech") is False

    def test_update_category(self, memory_system):
        memory_system.add_category("tech", "Technology")
        memory_system.update_category("tech", "Updated")
        assert memory_system.categories["tech"]["version"] == 2

    def test_categorize_memory(self, memory_system):
        memory_system.add_memory("test", categories=["tech"])
        memory_system.categorize_memory(0, ["personal"])
        assert set(memory_system.memories[0]["categories"]) == {"tech", "personal"}

    def test_add_relationship(self, memory_system):
        memory_system.add_category("programming")
        memory_system.add_category("tech")
        assert memory_system.add_relationship("programming", "part_of", "tech") is True

    def test_persistence(self, tmp_storage):
        store = JsonFileStore(tmp_storage)
        ms1 = MemorySystem(storage=store)
        ms1.add_memory("persisted", categories=["tech"])

        ms2 = MemorySystem(storage=JsonFileStore(tmp_storage))
        assert len(ms2.memories) == 1
        assert ms2.memories[0]["content"] == "persisted"

    def test_retrieve_by_similarity(self, populated_system):
        results = populated_system.retrieve_by_similarity("programming language", top_k=3)
        assert len(results) > 0
        assert results[0]["similarity"] > 0

    def test_retrieve_by_category(self, populated_system):
        results = populated_system.retrieve_by_category(["cooking"])
        assert len(results) == 3

    def test_hybrid_search(self, populated_system):
        results = populated_system.hybrid_search("deep learning", categories=["machine_learning"], top_k=3)
        assert len(results) > 0

    def test_enhanced_hybrid_search(self, populated_system):
        results = populated_system.enhanced_hybrid_search("programming language readability", top_k=5)
        assert len(results) > 0
        assert "retrieval_method" in results[0]

    def test_hnsw_index_persistence(self, tmp_storage):
        store = JsonFileStore(tmp_storage)
        ms1 = MemorySystem(storage=store)
        ms1.add_memory("test content for persistence")
        ms1.add_memory("another memory")

        assert os.path.exists(store._index_path)
        assert os.path.exists(store._mapping_path)

        ms2 = MemorySystem(storage=JsonFileStore(tmp_storage))
        assert ms2.vector_index.get_current_count() == 2

    def test_generate_statistics(self, populated_system):
        stats = populated_system.generate_statistics()
        assert stats["total_memories"] == 10
        assert stats["active_categories"] >= 3


# ─── EnhancedMemory Tests ─────────────────────────────────────────────────

class TestEnhancedMemory:

    def test_get_category_embedding(self, enhanced_memory):
        emb = enhanced_memory.get_category_embedding("programming")
        assert emb is not None
        assert isinstance(emb, np.ndarray)

    def test_get_category_embedding_nonexistent(self, enhanced_memory):
        assert enhanced_memory.get_category_embedding("nonexistent") is None

    def test_get_category_distribution(self, enhanced_memory):
        dist = enhanced_memory.get_category_distribution("programming")
        assert dist is not None
        assert dist["member_count"] >= 4
        assert dist["variance"] is not None

    def test_get_category_similarity(self, enhanced_memory):
        sim = enhanced_memory.get_category_similarity("programming", "machine_learning")
        assert sim is not None
        assert 0.0 <= sim <= 1.0
        sim_cooking = enhanced_memory.get_category_similarity("programming", "cooking")
        if sim_cooking is not None:
            assert sim > sim_cooking

    def test_suggest_categories_for_memory(self, enhanced_memory):
        mid = enhanced_memory.memory_system.add_memory(
            "Backpropagation computes gradients for training neural networks."
        )
        suggestions = enhanced_memory.suggest_categories_for_memory(mid, threshold=0.5)
        assert isinstance(suggestions, list)

    def test_discover_category_relationships(self, enhanced_memory):
        rels = enhanced_memory.discover_category_relationships(min_similarity=0.3)
        assert isinstance(rels, list)

    def test_find_category_outliers(self, enhanced_memory):
        outliers = enhanced_memory.find_category_outliers("programming", threshold=0.5)
        assert isinstance(outliers, list)

    def test_suggest_new_categories(self, enhanced_memory):
        ms = enhanced_memory.memory_system
        ms.add_memory("The stock market crashed today due to uncertainty.")
        ms.add_memory("Bond yields are rising as investors flee to safety.")
        ms.add_memory("Cryptocurrency prices dropped significantly overnight.")
        ms.add_memory("The Federal Reserve is considering interest rate changes.")
        suggestions = enhanced_memory.suggest_new_categories(min_memories=3, similarity_threshold=0.5)
        assert isinstance(suggestions, list)


# ─── MemoryFeedbackLoop Tests ─────────────────────────────────────────────

class TestMemoryFeedbackLoop:

    def test_chatbot_memory_search(self, feedback_loop):
        result = feedback_loop.chatbot_memory_search("programming language", result_count=3)
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_chatbot_memory_search_enhanced(self, feedback_loop):
        result = feedback_loop.chatbot_memory_search("deep learning", result_count=3, use_enhanced_search=True)
        assert "results" in result

    def test_search_pattern_tracking(self, feedback_loop):
        feedback_loop.chatbot_memory_search("programming language features")
        feedback_loop.chatbot_memory_search("programming best practices")
        assert feedback_loop.search_patterns.get("programming", 0) >= 2

    def test_update_quality_metrics(self, feedback_loop):
        metrics = feedback_loop.update_quality_metrics()
        assert isinstance(metrics, dict)
        if metrics:
            assert metrics["system_metrics"]["total_memories"] == 10

    def test_suggest_improvements(self, feedback_loop):
        feedback_loop.chatbot_memory_search("programming")
        feedback_loop.chatbot_memory_search("programming language")
        suggestions = feedback_loop.suggest_improvements()
        assert "merge_suggestions" in suggestions

    def test_log_category_feedback(self, feedback_loop):
        feedback_loop.log_category_feedback("programming", True, memory_id=0)
        events = [e for e in feedback_loop.feedback_history["feedback_events"] if e["type"] == "category_feedback"]
        assert len(events) >= 1

    def test_log_memory_categorization(self, feedback_loop):
        feedback_loop.log_memory_categorization(0, ["programming", "tech"], ["programming"])
        events = [e for e in feedback_loop.feedback_history["feedback_events"] if e["type"] == "categorization_feedback"]
        assert events[-1]["acceptance_rate"] == 0.5


# ─── SynapticCore Facade Tests ────────────────────────────────────────────

class TestSynapticCoreFacade:

    def test_facade_creates_subsystems(self, core):
        assert core.memory is not None
        assert core.enhanced is not None
        assert core.feedback is not None
        assert core.storage is not None

    def test_facade_round_trip(self, core):
        core.memory.add_memory("Test memory about programming", categories=["tech"])
        core.memory.add_memory("Another memory about cooking pasta", categories=["food"])

        results = core.memory.retrieve_by_similarity("programming", top_k=2)
        assert len(results) > 0

        suggestions = core.enhanced.suggest_categories_for_memory(0, threshold=0.3)
        assert isinstance(suggestions, list)

    def test_facade_feedback_integration(self, core):
        core.memory.add_memory("Python programming is fun", categories=["tech"])
        core.memory.add_memory("JavaScript runs in browsers", categories=["tech"])
        core.memory.add_memory("Machine learning uses data", categories=["ml"])

        result = core.feedback.chatbot_memory_search("programming", result_count=2)
        assert "results" in result


# ─── LLM Provider Tests ───────────────────────────────────────────────────

class TestLLMProvider:

    def test_import(self):
        from synapticcore.llm import LLMProvider, create_provider
        assert LLMProvider is not None

    def test_create_provider_unknown(self):
        from synapticcore.llm import create_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent")
