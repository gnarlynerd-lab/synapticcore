"""Shared test fixtures for synapticcore package tests."""

import pytest
from synapticcore import SynapticCore, MemorySystem, EnhancedMemory, MemoryFeedbackLoop
from synapticcore.storage import JsonFileStore


@pytest.fixture
def tmp_storage(tmp_path):
    return str(tmp_path / "test_memory.json")


@pytest.fixture
def tmp_feedback(tmp_path):
    return str(tmp_path / "test_feedback.json")


@pytest.fixture
def memory_system(tmp_storage):
    store = JsonFileStore(tmp_storage)
    return MemorySystem(storage=store)


@pytest.fixture
def populated_system(memory_system):
    ms = memory_system
    ms.add_category("programming", "Computer programming and software development")
    ms.add_category("machine_learning", "Machine learning and AI techniques")
    ms.add_category("cooking", "Food preparation and recipes")

    ms.add_memory("Python is a high-level programming language known for readability.", categories=["programming"])
    ms.add_memory("JavaScript is essential for web development and runs in browsers.", categories=["programming"])
    ms.add_memory("Rust provides memory safety without garbage collection.", categories=["programming"])
    ms.add_memory("Software design patterns help create maintainable code.", categories=["programming"])
    ms.add_memory("Neural networks learn patterns from training data.", categories=["machine_learning"])
    ms.add_memory("Gradient descent optimizes model parameters iteratively.", categories=["machine_learning"])
    ms.add_memory("TensorFlow and PyTorch are popular deep learning frameworks.", categories=["programming", "machine_learning"])
    ms.add_memory("Sourdough bread requires a fermented starter and long rise time.", categories=["cooking"])
    ms.add_memory("Caramelizing onions takes low heat and patience for best flavor.", categories=["cooking"])
    ms.add_memory("Mise en place means preparing all ingredients before cooking.", categories=["cooking"])
    return ms


@pytest.fixture
def enhanced_memory(populated_system):
    return EnhancedMemory(populated_system)


@pytest.fixture
def feedback_loop(populated_system, enhanced_memory, tmp_feedback):
    return MemoryFeedbackLoop(populated_system, enhanced_memory=enhanced_memory, feedback_log_path=tmp_feedback)


@pytest.fixture
def core(tmp_storage):
    return SynapticCore(storage_path=tmp_storage)
