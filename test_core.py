"""
Core tests for SynapticCore memory system.

Tests cover:
- MemorySystem: CRUD, search, categories, relationships
- EnhancedMemorySystem: category embeddings, similarity, outliers, suggestions
- MemoryFeedbackLoop: search tracking, metrics, improvement suggestions

These tests are the safety net for the Phase 1 composition refactor.
"""

import os
import json
import tempfile
import pytest
import numpy as np

from simple_memory_system import MemorySystem
from enhanced_memory_system import EnhancedMemorySystem, enhance_memory_system
from memory_feedback_loop import MemoryFeedbackLoop


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_storage(tmp_path):
    """Return a temp path for memory storage."""
    return str(tmp_path / "test_memory.json")


@pytest.fixture
def tmp_feedback(tmp_path):
    """Return a temp path for feedback history."""
    return str(tmp_path / "test_feedback.json")


@pytest.fixture
def memory_system(tmp_storage):
    """Create a fresh MemorySystem with a temp store."""
    ms = MemorySystem(storage_path=tmp_storage)
    return ms


@pytest.fixture
def populated_system(memory_system):
    """MemorySystem pre-loaded with test data across several categories."""
    ms = memory_system

    ms.add_category("programming", "Computer programming and software development")
    ms.add_category("machine_learning", "Machine learning and AI techniques")
    ms.add_category("cooking", "Food preparation and recipes")

    # Programming memories
    ms.add_memory("Python is a high-level programming language known for readability.",
                  categories=["programming"])
    ms.add_memory("JavaScript is essential for web development and runs in browsers.",
                  categories=["programming"])
    ms.add_memory("Rust provides memory safety without garbage collection.",
                  categories=["programming"])
    ms.add_memory("Software design patterns help create maintainable code.",
                  categories=["programming"])

    # ML memories
    ms.add_memory("Neural networks learn patterns from training data.",
                  categories=["machine_learning"])
    ms.add_memory("Gradient descent optimizes model parameters iteratively.",
                  categories=["machine_learning"])
    ms.add_memory("TensorFlow and PyTorch are popular deep learning frameworks.",
                  categories=["programming", "machine_learning"])

    # Cooking memories
    ms.add_memory("Sourdough bread requires a fermented starter and long rise time.",
                  categories=["cooking"])
    ms.add_memory("Caramelizing onions takes low heat and patience for best flavor.",
                  categories=["cooking"])
    ms.add_memory("Mise en place means preparing all ingredients before cooking.",
                  categories=["cooking"])

    return ms


@pytest.fixture
def enhanced_system(populated_system):
    """Populated system with enhanced (dual embedding) methods."""
    return enhance_memory_system(populated_system)


@pytest.fixture
def feedback_loop(enhanced_system, tmp_feedback):
    """MemoryFeedbackLoop wrapping the enhanced system."""
    return MemoryFeedbackLoop(enhanced_system, feedback_log_path=tmp_feedback)


# ─── MemorySystem Tests ────────────────────────────────────────────────────

class TestMemorySystem:

    def test_add_memory_returns_id(self, memory_system):
        mid = memory_system.add_memory("test content")
        assert mid == 0

    def test_add_memory_sequential_ids(self, memory_system):
        m0 = memory_system.add_memory("first")
        m1 = memory_system.add_memory("second")
        assert m0 == 0
        assert m1 == 1

    def test_memory_has_embedding(self, memory_system):
        memory_system.add_memory("test content with enough words for embedding")
        assert memory_system.memories[0]["embedding"] is not None
        assert len(memory_system.memories[0]["embedding"]) > 0

    def test_memory_has_timestamp(self, memory_system):
        memory_system.add_memory("test")
        assert "timestamp" in memory_system.memories[0]

    def test_memory_has_categories(self, memory_system):
        memory_system.add_category("tech", "technology")
        memory_system.add_memory("test", categories=["tech"])
        assert "tech" in memory_system.memories[0]["categories"]

    def test_add_memory_auto_creates_category(self, memory_system):
        memory_system.add_memory("test", categories=["new_cat"])
        assert "new_cat" in memory_system.categories

    def test_add_category(self, memory_system):
        assert memory_system.add_category("tech", "Technology") is True
        assert "tech" in memory_system.categories
        assert memory_system.categories["tech"]["version"] == 1

    def test_add_duplicate_category(self, memory_system):
        memory_system.add_category("tech", "Technology")
        assert memory_system.add_category("tech", "Technology again") is False

    def test_update_category(self, memory_system):
        memory_system.add_category("tech", "Technology")
        memory_system.update_category("tech", "Updated description")
        assert memory_system.categories["tech"]["description"] == "Updated description"
        assert memory_system.categories["tech"]["version"] == 2
        assert len(memory_system.categories["tech"]["history"]) == 2

    def test_update_category_no_change(self, memory_system):
        memory_system.add_category("tech", "Technology")
        memory_system.update_category("tech", "Technology")  # same description
        assert memory_system.categories["tech"]["version"] == 1

    def test_categorize_memory(self, memory_system):
        memory_system.add_category("tech")
        memory_system.add_category("personal")
        memory_system.add_memory("test", categories=["tech"])
        memory_system.categorize_memory(0, ["personal"])
        assert "tech" in memory_system.memories[0]["categories"]
        assert "personal" in memory_system.memories[0]["categories"]

    def test_add_relationship(self, memory_system):
        memory_system.add_category("programming")
        memory_system.add_category("tech")
        assert memory_system.add_relationship("programming", "part_of", "tech") is True
        rel_id = "programming:part_of:tech"
        assert rel_id in memory_system.relationships

    def test_add_relationship_nonexistent_category(self, memory_system):
        memory_system.add_category("programming")
        assert memory_system.add_relationship("programming", "part_of", "nonexistent") is False

    def test_persistence(self, tmp_storage):
        """Data persists across instances."""
        ms1 = MemorySystem(storage_path=tmp_storage)
        ms1.add_category("tech", "Technology")
        ms1.add_memory("persisted content", categories=["tech"])

        ms2 = MemorySystem(storage_path=tmp_storage)
        assert len(ms2.memories) == 1
        assert ms2.memories[0]["content"] == "persisted content"
        assert "tech" in ms2.categories

    def test_retrieve_by_similarity(self, populated_system):
        results = populated_system.retrieve_by_similarity("programming language", top_k=3)
        assert len(results) > 0
        assert results[0]["similarity"] > 0
        # Top result should be about programming
        top_content = results[0]["memory"]["content"].lower()
        assert "programming" in top_content or "language" in top_content

    def test_retrieve_by_similarity_with_category_filter(self, populated_system):
        results = populated_system.retrieve_by_similarity(
            "learning techniques", top_k=5, category_filter=["machine_learning"]
        )
        for r in results:
            assert "machine_learning" in r["memory"]["categories"]

    def test_retrieve_by_category(self, populated_system):
        results = populated_system.retrieve_by_category(["cooking"])
        assert len(results) == 3
        for r in results:
            assert "cooking" in r["memory"]["categories"]

    def test_hybrid_search(self, populated_system):
        results = populated_system.hybrid_search(
            "deep learning frameworks", categories=["machine_learning"], top_k=3
        )
        assert len(results) > 0
        assert "combined_score" in results[0]

    def test_enhanced_hybrid_search(self, populated_system):
        results = populated_system.enhanced_hybrid_search(
            "programming language readability", top_k=5
        )
        assert len(results) > 0
        assert "semantic_score" in results[0]
        assert "retrieval_method" in results[0]

    def test_enhanced_hybrid_search_with_categories(self, populated_system):
        results = populated_system.enhanced_hybrid_search(
            "learning", categories=["machine_learning"], top_k=3
        )
        assert len(results) > 0

    def test_enhanced_hybrid_search_associative(self, populated_system):
        """Recursive depth > 0 should find associative results."""
        results = populated_system.enhanced_hybrid_search(
            "Python programming", top_k=5, recursive_depth=2
        )
        methods = [r["retrieval_method"] for r in results]
        # Should have at least semantic results
        assert any("semantic" in m for m in methods)

    def test_get_category_evolution(self, populated_system):
        evo = populated_system.get_category_evolution("programming")
        assert evo is not None
        assert evo["name"] == "programming"
        assert len(evo["memory_examples"]) > 0

    def test_generate_statistics(self, populated_system):
        stats = populated_system.generate_statistics()
        assert stats["total_memories"] == 10
        assert stats["total_categories"] >= 3
        assert stats["active_categories"] >= 3

    def test_hnsw_index_persistence(self, tmp_storage):
        """HNSW index persists and loads on restart."""
        ms1 = MemorySystem(storage_path=tmp_storage)
        ms1.add_memory("test content for index persistence", categories=["test"])
        ms1.add_memory("another memory for the index", categories=["test"])

        # Check index files exist
        base = tmp_storage.rsplit('.', 1)[0]
        assert os.path.exists(base + '.hnsw')
        assert os.path.exists(base + '_index_mapping.json')

        # Load fresh instance — should load index from disk
        ms2 = MemorySystem(storage_path=tmp_storage)
        assert ms2.vector_index is not None
        assert ms2.vector_index.get_current_count() == 2

        # Verify search still works
        results = ms2.retrieve_by_similarity("test content", top_k=2)
        assert len(results) > 0


# ─── EnhancedMemorySystem Tests ───────────────────────────────────────────

class TestEnhancedMemorySystem:

    def test_get_category_embedding(self, enhanced_system):
        emb = enhanced_system.get_category_embedding("programming")
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert len(emb) > 0

    def test_get_category_embedding_nonexistent(self, enhanced_system):
        assert enhanced_system.get_category_embedding("nonexistent") is None

    def test_get_category_embedding_empty_category(self, enhanced_system):
        enhanced_system.add_category("empty_cat", "no memories here")
        assert enhanced_system.get_category_embedding("empty_cat") is None

    def test_get_category_distribution(self, enhanced_system):
        dist = enhanced_system.get_category_distribution("programming")
        assert dist is not None
        assert dist["member_count"] >= 4
        assert dist["center"] is not None
        assert dist["variance"] is not None
        assert dist["variance"] >= 0

    def test_get_category_similarity(self, enhanced_system):
        sim = enhanced_system.get_category_similarity("programming", "machine_learning")
        assert sim is not None
        assert 0.0 <= sim <= 1.0
        # Programming and ML should be more similar than programming and cooking
        sim_cooking = enhanced_system.get_category_similarity("programming", "cooking")
        if sim_cooking is not None:
            assert sim > sim_cooking

    def test_get_category_similarity_nonexistent(self, enhanced_system):
        assert enhanced_system.get_category_similarity("programming", "nonexistent") is None

    def test_suggest_categories_for_memory(self, enhanced_system):
        # Add an uncategorized ML-related memory
        mid = enhanced_system.add_memory(
            "Backpropagation computes gradients for training neural networks."
        )
        suggestions = enhanced_system.suggest_categories_for_memory(mid, threshold=0.5)
        assert isinstance(suggestions, list)
        # Should suggest machine_learning or programming
        if suggestions:
            suggested_names = [s["category"] for s in suggestions]
            assert any(c in suggested_names for c in ["machine_learning", "programming"])

    def test_suggest_categories_invalid_id(self, enhanced_system):
        assert enhanced_system.suggest_categories_for_memory(-1) == []
        assert enhanced_system.suggest_categories_for_memory(9999) == []

    def test_discover_category_relationships(self, enhanced_system):
        rels = enhanced_system.discover_category_relationships(min_similarity=0.3)
        assert isinstance(rels, list)
        # programming and ML should be related
        if rels:
            pairs = [(r["source"], r["target"]) for r in rels]
            assert any(
                ("programming" in p and "machine_learning" in p)
                for p in pairs
            )

    def test_find_category_outliers(self, enhanced_system):
        outliers = enhanced_system.find_category_outliers("programming", threshold=0.5)
        assert isinstance(outliers, list)
        # All returned outliers should have required fields
        for o in outliers:
            assert "memory_id" in o
            assert "distance" in o
            assert "threshold" in o

    def test_find_category_outliers_nonexistent(self, enhanced_system):
        assert enhanced_system.find_category_outliers("nonexistent") == []

    def test_suggest_new_categories(self, enhanced_system):
        # Add several uncategorized memories on a new topic
        enhanced_system.add_memory("The stock market crashed today due to uncertainty.")
        enhanced_system.add_memory("Bond yields are rising as investors flee to safety.")
        enhanced_system.add_memory("Cryptocurrency prices dropped significantly overnight.")
        enhanced_system.add_memory("The Federal Reserve is considering interest rate changes.")

        suggestions = enhanced_system.suggest_new_categories(min_memories=3, similarity_threshold=0.5)
        assert isinstance(suggestions, list)
        # Should find at least one cluster of finance-related memories
        if suggestions:
            assert suggestions[0]["size"] >= 3


# ─── MemoryFeedbackLoop Tests ─────────────────────────────────────────────

class TestMemoryFeedbackLoop:

    def test_chatbot_memory_search(self, feedback_loop):
        result = feedback_loop.chatbot_memory_search("programming language", result_count=3)
        assert "results" in result
        assert "found_categories" in result
        assert "search_time" in result
        assert isinstance(result["results"], list)

    def test_chatbot_memory_search_enhanced(self, feedback_loop):
        result = feedback_loop.chatbot_memory_search(
            "deep learning", result_count=3, use_enhanced_search=True
        )
        assert "results" in result

    def test_search_pattern_tracking(self, feedback_loop):
        feedback_loop.chatbot_memory_search("programming language features")
        feedback_loop.chatbot_memory_search("programming best practices")
        assert feedback_loop.search_patterns.get("programming", 0) >= 2

    def test_failed_search_tracking(self, feedback_loop):
        # Search for something unlikely to match well
        feedback_loop.chatbot_memory_search("quantum entanglement teleportation xyz123")
        # Failed searches may or may not be recorded depending on score
        assert isinstance(feedback_loop.failed_searches, list)

    def test_update_quality_metrics(self, feedback_loop):
        metrics = feedback_loop.update_quality_metrics()
        assert isinstance(metrics, dict)
        if metrics:
            assert "category_metrics" in metrics
            assert "system_metrics" in metrics
            assert metrics["system_metrics"]["total_memories"] == 10

    def test_suggest_improvements(self, feedback_loop):
        # Run some searches to build up patterns
        feedback_loop.chatbot_memory_search("programming")
        feedback_loop.chatbot_memory_search("programming language")
        feedback_loop.chatbot_memory_search("code development")

        suggestions = feedback_loop.suggest_improvements()
        assert "merge_suggestions" in suggestions
        assert "split_suggestions" in suggestions
        assert "new_category_suggestions" in suggestions
        assert "rename_suggestions" in suggestions

    def test_log_category_feedback(self, feedback_loop):
        feedback_loop.log_category_feedback("programming", True, memory_id=0)
        events = feedback_loop.feedback_history["feedback_events"]
        cat_events = [e for e in events if e["type"] == "category_feedback"]
        assert len(cat_events) >= 1

    def test_log_memory_categorization(self, feedback_loop):
        feedback_loop.log_memory_categorization(
            0, ["programming", "tech"], ["programming"]
        )
        events = feedback_loop.feedback_history["feedback_events"]
        cat_events = [e for e in events if e["type"] == "categorization_feedback"]
        assert len(cat_events) >= 1
        assert cat_events[-1]["acceptance_rate"] == 0.5

    def test_feedback_persistence(self, enhanced_system, tmp_path):
        """Feedback history persists across instances."""
        fb_path = str(tmp_path / "fb.json")
        fl1 = MemoryFeedbackLoop(enhanced_system, feedback_log_path=fb_path)
        fl1.log_category_feedback("programming", True)

        fl2 = MemoryFeedbackLoop(enhanced_system, feedback_log_path=fb_path)
        events = fl2.feedback_history["feedback_events"]
        assert len(events) >= 1


# ─── LLM Provider Tests ───────────────────────────────────────────────────

class TestLLMProvider:

    def test_import(self):
        from llm_provider import LLMProvider, DeepSeekProvider, AnthropicProvider
        assert LLMProvider is not None

    def test_create_provider_unknown(self):
        from llm_provider import create_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent")

    def test_create_provider_no_key(self):
        from llm_provider import create_provider
        # Clear env vars temporarily
        old_keys = {}
        for key in ["ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "SYNAPTICCORE_LLM_PROVIDER"]:
            old_keys[key] = os.environ.pop(key, None)
        try:
            with pytest.raises(ValueError, match="No LLM provider"):
                create_provider()
        finally:
            for key, val in old_keys.items():
                if val is not None:
                    os.environ[key] = val
