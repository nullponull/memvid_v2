# tests/test_integration.py
import pytest
import os
from pathlib import Path
import tempfile
import shutil # For easier cleanup of directories

# Adjust import path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memvid import MemvidEncoder, MemvidRetriever, MemvidChat
from memvid.config import get_default_config
from unittest.mock import patch # For OPENAI_AVAILABLE

# Sample data for testing
SAMPLE_CHUNKS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is rapidly evolving.",
    "SQLite is a lightweight, file-based database.",
    "FAISS provides efficient similarity search for vectors.",
    "Memvid aims to store and retrieve text efficiently.",
    "Unit testing ensures individual components work correctly.",
    "Integration testing checks if components work together.",
    "The lazy dog was not amused by the quick brown fox."
]

@pytest.fixture
def temp_output_dir():
    """Creates a temporary directory for test outputs and cleans up afterwards."""
    with tempfile.TemporaryDirectory(prefix="memvid_integration_test_") as tmpdir:
        yield Path(tmpdir)

def test_full_workflow_encode_retrieve(temp_output_dir):
    """Test encoding data, building memory, and retrieving it."""
    db_path = temp_output_dir / "test_memory.db"
    index_prefix = temp_output_dir / "test_memory_idx"

    # 1. Configure and Initialize Encoder
    test_config = get_default_config()
    test_config["database"]["path"] = str(db_path)
    # Small batch size for faster processing in tests if many chunks were added
    test_config["retrieval"]["batch_size"] = 2

    encoder = MemvidEncoder(config=test_config)

    # 2. Add Data
    encoder.add_chunks(SAMPLE_CHUNKS)
    assert encoder.db_manager.get_total_chunks() == len(SAMPLE_CHUNKS)

    # 3. Build Memory (DB is populated, FAISS index is built and saved)
    build_stats = encoder.build_memory(index_file_path_prefix=str(index_prefix))

    assert (index_prefix.with_suffix(".faiss")).exists()
    assert (index_prefix.with_suffix(".indexinfo.json")).exists()
    assert build_stats["total_chunks_in_db"] == len(SAMPLE_CHUNKS)
    assert build_stats["index_stats"]["total_indexed_chunks"] == len(SAMPLE_CHUNKS)

    # 4. Initialize Retriever
    # Config for retriever should also point to the same DB path
    retriever_config = get_default_config()
    retriever_config["database"]["path"] = str(db_path)
    retriever = MemvidRetriever(index_file_path_prefix=str(index_prefix), config=retriever_config)

    # 5. Perform Searches
    results_fox = retriever.search("quick fox", top_k=2)
    assert len(results_fox) > 0
    assert any("fox" in r.lower() for r in results_fox)

    results_db = retriever.search_with_metadata("lightweight database", top_k=1)
    assert len(results_db) == 1
    assert "sqlite" in results_db[0]["text"].lower()
    assert results_db[0]["chunk_id"] > 0

    memvid_query_res = retriever.search_with_metadata("Memvid efficient storage", top_k=1)
    assert len(memvid_query_res) == 1
    memvid_chunk_id = memvid_query_res[0]["chunk_id"]

    retrieved_chunk_text = retriever.get_chunk_by_id(memvid_chunk_id)
    assert retrieved_chunk_text is not None
    assert "memvid aims" in retrieved_chunk_text.lower()

    assert retriever.get_chunk_by_id(99999) is None


def test_integration_with_chat_context_only(temp_output_dir):
    """Test integration with MemvidChat (context-only, no LLM calls)."""
    db_path = temp_output_dir / "chat_test.db"
    index_prefix = temp_output_dir / "chat_test_idx"

    # 1. Encode and Build Memory
    encoder_config = get_default_config()
    encoder_config["database"]["path"] = str(db_path)
    # Use smaller context chunks for chat test to make assertions easier
    encoder_config["chat"]["context_chunks"] = 2

    encoder = MemvidEncoder(config=encoder_config)
    chat_sample_chunks = [
        "Blue is a primary color.",
        "The sky is often perceived as blue due to Rayleigh scattering.",
        "Red is another primary color, often associated with warmth."
    ]
    encoder.add_chunks(chat_sample_chunks)
    encoder.build_memory(index_file_path_prefix=str(index_prefix))

    # 2. Initialize MemvidChat (no API key for context-only)
    chat_config = get_default_config()
    chat_config["database"]["path"] = str(db_path)
    chat_config["chat"]["context_chunks"] = 2 # Match encoder config or set as desired for chat

    # Patch OPENAI_AVAILABLE to ensure no LLM calls for this test
    with patch('memvid.chat.OPENAI_AVAILABLE', False):
        chat = MemvidChat(index_file_path_prefix=str(index_prefix), config=chat_config)
        chat.start_session()

        response = chat.chat("Tell me about blue.")
        # Default context-only response format is "Based on the knowledge base..."
        assert "Based on the knowledge base" in response
        assert "sky is often perceived as blue" in response.lower()
        assert "blue is a primary color" in response.lower()
        # "Red is another primary color" should not be in the top 2 for "Tell me about blue"
        assert "red is another primary color" not in response.lower()

    # 3. Test search_context via chat
    # Re-init chat without the patch for OPENAI_AVAILABLE (it doesn't affect search_context)
    chat_no_patch_config = get_default_config() # Fresh config
    chat_no_patch_config["database"]["path"] = str(db_path)
    chat_no_patch = MemvidChat(index_file_path_prefix=str(index_prefix), config=chat_no_patch_config)

    context_results = chat_no_patch.search_context("information about red", top_k=1)
    assert len(context_results) == 1
    assert "red is another primary color" in context_results[0]["text"].lower()
