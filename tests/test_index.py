# tests/test_index.py
import pytest
import numpy as np
import faiss # For creating sample indexes or checking types
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Adjust import path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memvid.index import IndexManager
from memvid.database import DatabaseManager # For type hinting and mocking its instance

# Default config for testing (can be overridden in specific tests)
TEST_CONFIG_FLAT = {
    "embedding": {"model": "all-MiniLM-L6-v2", "dimension": 384}, # Ensure this matches a real model for dim
    "index": {"type": "Flat"},
    "retrieval": {"batch_size": 2} # Small batch size for testing
}
TEST_CONFIG_IVF = {
    "embedding": {"model": "all-MiniLM-L6-v2", "dimension": 384},
    "index": {"type": "IVF", "nlist": 2}, # Small nlist for testing
    "retrieval": {"batch_size": 2}
}


@pytest.fixture
def mock_db_manager():
    """Fixture for a mocked DatabaseManager."""
    mock_db = Mock(spec=DatabaseManager)
    # Setup common return values if needed by most tests, e.g.
    # mock_db.get_all_chunks_for_indexing.return_value = []
    return mock_db

@pytest.fixture
def temp_index_prefix(tmp_path):
    """Provides a temporary path prefix for saving/loading index files."""
    return tmp_path / "test_idx"


def test_index_manager_initialization_flat():
    manager = IndexManager(config=TEST_CONFIG_FLAT)
    assert manager.dimension == TEST_CONFIG_FLAT["embedding"]["dimension"]
    assert isinstance(manager.index, faiss.IndexIDMap)
    assert isinstance(manager.index.index, faiss.IndexFlatL2) # Underlying index for IndexIDMap
    assert manager.index.ntotal == 0

def test_index_manager_initialization_ivf():
    manager = IndexManager(config=TEST_CONFIG_IVF)
    assert manager.dimension == TEST_CONFIG_IVF["embedding"]["dimension"]
    assert isinstance(manager.index, faiss.IndexIDMap)
    assert isinstance(manager.index.index, faiss.IndexIVFFlat)
    assert not manager.index.index.is_trained # IVF needs training (access .index for actual IVF index)

def test_clear_index(mock_db_manager): # mock_db_manager not strictly needed here but good for consistency
    manager = IndexManager(config=TEST_CONFIG_FLAT)
    # Simulate adding some data (simplified, real add is via build_index_from_db)
    sample_embeddings = np.random.rand(5, manager.dimension).astype('float32')
    sample_ids = np.arange(5).astype('int64')
    manager.index.add_with_ids(sample_embeddings, sample_ids)
    assert manager.index.ntotal == 5

    manager.clear_index()
    assert manager.index.ntotal == 0

    # For IVF, it should also reset the trained state.
    ivf_config = {**TEST_CONFIG_IVF} # Make a copy to avoid modifying global
    ivf_manager = IndexManager(config=ivf_config)
    # Ensure enough samples for training if nlist is small.
    # IVF training requires at least `nlist` distinct vectors, ideally more.
    # For `nlist=2`, we need at least 2, let's provide more.
    num_train_samples = ivf_config["index"]["nlist"] * 10
    train_embeddings = np.random.rand(num_train_samples, ivf_manager.dimension).astype('float32')

    if not ivf_manager.index.index.is_trained: # Access .index for actual IVF index
         ivf_manager.index.index.train(train_embeddings)
    ivf_manager.index.add_with_ids(train_embeddings[:5], np.arange(5).astype('int64')) # Add some IDs
    assert ivf_manager.index.index.is_trained
    assert ivf_manager.index.ntotal == 5

    ivf_manager.clear_index()
    assert not ivf_manager.index.index.is_trained # Check underlying index
    assert ivf_manager.index.ntotal == 0


@patch('sentence_transformers.SentenceTransformer') # Mock the actual embedding model
def test_build_index_from_db_flat(MockSentenceTransformer, mock_db_manager, temp_index_prefix):
    mock_model_instance = MockSentenceTransformer.return_value
    # Simulate embedding output: needs to be (num_chunks, dimension)
    mock_model_instance.encode.return_value = np.random.rand(2, TEST_CONFIG_FLAT["embedding"]["dimension"]).astype('float32')

    manager = IndexManager(config=TEST_CONFIG_FLAT)

    # Mock DB returning chunks
    test_chunks_from_db = [(1, "text one"), (2, "text two")]
    mock_db_manager.get_all_chunks_for_indexing.return_value = test_chunks_from_db

    manager.build_index_from_db(mock_db_manager, show_progress=False)

    mock_db_manager.get_all_chunks_for_indexing.assert_called_once()
    mock_model_instance.encode.assert_called_once_with(["text one", "text two"], show_progress_bar=False, batch_size=TEST_CONFIG_FLAT["retrieval"]["batch_size"])
    assert manager.index.ntotal == 2
    mock_db_manager.set_info.assert_called_with("total_indexed_chunks", 2)

@patch('sentence_transformers.SentenceTransformer')
def test_build_index_from_db_ivf_training(MockSentenceTransformer, mock_db_manager):
    mock_model_instance = MockSentenceTransformer.return_value

    ivf_config = {**TEST_CONFIG_IVF} # Use a copy
    num_samples = ivf_config["index"]["nlist"] * 5
    sample_embeddings = np.random.rand(num_samples, ivf_config["embedding"]["dimension"]).astype('float32')
    mock_model_instance.encode.return_value = sample_embeddings

    manager = IndexManager(config=ivf_config)
    test_chunks_from_db = [(i, f"text {i}") for i in range(num_samples)]
    mock_db_manager.get_all_chunks_for_indexing.return_value = test_chunks_from_db

    manager.build_index_from_db(mock_db_manager, show_progress=False)

    assert manager.index.index.is_trained # Access .index for actual IVF index
    assert manager.index.ntotal == num_samples
    mock_db_manager.set_info.assert_called_with("total_indexed_chunks", num_samples)


def test_search_empty_index():
    manager = IndexManager(config=TEST_CONFIG_FLAT)
    results = manager.search("query", top_k=3)
    assert results == []

@patch('sentence_transformers.SentenceTransformer')
def test_search_populated_index(MockSentenceTransformer, mock_db_manager): # mock_db_manager for build
    mock_model_instance = MockSentenceTransformer.return_value

    small_dim_config = {**TEST_CONFIG_FLAT, "embedding": {**TEST_CONFIG_FLAT["embedding"], "dimension": 2}}
    manager = IndexManager(config=small_dim_config)

    # Embeddings for building the index (3 vectors of dimension 2)
    build_embeddings = np.array([[0.1, 0.2], [0.8, 0.9], [0.3, 0.35]], dtype='float32')
    # Embedding for the search query
    query_embedding = np.array([[0.11, 0.21]], dtype='float32') # Closest to [0.1, 0.2]

    mock_model_instance.encode.side_effect = [
        build_embeddings, # For build_index_from_db call
        query_embedding   # For search call
    ]

    test_chunks_from_db = [(10, "text A"), (20, "text B"), (30, "text C")]
    mock_db_manager.get_all_chunks_for_indexing.return_value = test_chunks_from_db
    manager.build_index_from_db(mock_db_manager, show_progress=False)

    assert manager.index.ntotal == 3

    results = manager.search("query for A", top_k=1)

    assert len(results) == 1
    assert results[0][0] == 10 # chunk_id of "text A" (vector [0.1, 0.2] is closest to query [0.11, 0.21])
    assert isinstance(results[0][1], float) # distance

def test_save_and_load_index(mock_db_manager, temp_index_prefix): # mock_db for build
    manager_orig = IndexManager(config=TEST_CONFIG_FLAT)

    # Simulate building an index
    with patch.object(manager_orig, 'embedding_model') as mock_embed_model:
        mock_embed_model.encode.return_value = np.random.rand(5, TEST_CONFIG_FLAT["embedding"]["dimension"]).astype('float32')
        test_chunks = [(i, f"text {i}") for i in range(5)]
        mock_db_manager.get_all_chunks_for_indexing.return_value = test_chunks
        manager_orig.build_index_from_db(mock_db_manager, show_progress=False)

    assert manager_orig.index.ntotal == 5
    manager_orig.save(str(temp_index_prefix))

    assert (temp_index_prefix.with_suffix('.faiss')).exists()
    assert (temp_index_prefix.with_suffix('.indexinfo.json')).exists()

    manager_loaded = IndexManager(config=TEST_CONFIG_FLAT)
    manager_loaded.load(str(temp_index_prefix))

    assert manager_loaded.index.ntotal == 5

    with open(temp_index_prefix.with_suffix('.indexinfo.json'), 'r') as f:
        info = json.load(f)
        assert info["total_indexed_chunks"] == 5
        assert info["config"]["embedding_model"] == TEST_CONFIG_FLAT["embedding"]["model"]

def test_load_non_existent_index(temp_index_prefix):
    manager = IndexManager(config=TEST_CONFIG_FLAT)
    manager.load(str(temp_index_prefix))
    assert manager.index.ntotal == 0
    assert not (temp_index_prefix.with_suffix('.faiss')).exists()

def test_get_stats(mock_db_manager):
    manager = IndexManager(config=TEST_CONFIG_FLAT)
    stats_empty = manager.get_stats()
    assert stats_empty["total_indexed_chunks"] == 0
    assert stats_empty["index_type"] == "Flat"
    assert stats_empty["is_trained"] == "N/A (Flat index or no index)"

    # Simulate build for populated stats
    with patch.object(manager, 'embedding_model') as mock_embed_model:
        mock_embed_model.encode.return_value = np.random.rand(3, manager.dimension).astype('float32')
        mock_db_manager.get_all_chunks_for_indexing.return_value = [(i,f"t{i}") for i in range(3)]
        manager.build_index_from_db(mock_db_manager)

    stats_populated = manager.get_stats()
    assert stats_populated["total_indexed_chunks"] == 3
    assert stats_populated["is_trained"] == "N/A (Flat index or no index)"

    # Test IVF stats
    ivf_manager = IndexManager(config=TEST_CONFIG_IVF)
    ivf_stats_empty = ivf_manager.get_stats()
    assert ivf_stats_empty["is_trained"] == False # Before training

    with patch.object(ivf_manager, 'embedding_model') as mock_embed_model_ivf:
        num_samples_ivf = TEST_CONFIG_IVF["index"]["nlist"] * 5
        mock_embed_model_ivf.encode.return_value = np.random.rand(num_samples_ivf, ivf_manager.dimension).astype('float32')
        mock_db_manager.reset_mock() # Reset for this new manager
        mock_db_manager.get_all_chunks_for_indexing.return_value = [(i,f"t_ivf_{i}") for i in range(num_samples_ivf)]
        ivf_manager.build_index_from_db(mock_db_manager)

    ivf_stats_populated = ivf_manager.get_stats()
    assert ivf_stats_populated["total_indexed_chunks"] == num_samples_ivf
    assert ivf_stats_populated["is_trained"] == True
