import pytest
from unittest.mock import Mock, patch, call
import os
from pathlib import Path
import tempfile # Not strictly needed for these unit tests but often useful

from memvid.retriever import MemvidRetriever
from memvid.config import get_default_config # For default config structure

# Helper to create a default-like config for tests
def create_test_config():
    config = get_default_config()
    # Override DB path for predictability in tests if needed
    config["database"]["path"] = "dummy/test_retriever_memory.db"
    config["retrieval"]["cache_size"] = 128 # Example, used by lru_cache
    return config

@patch('memvid.retriever.IndexManager')
@patch('memvid.retriever.DatabaseManager')
def test_retriever_initialization(MockDatabaseManager, MockIndexManager):
    mock_db_instance = MockDatabaseManager.return_value
    mock_index_instance = MockIndexManager.return_value
    
    test_config = create_test_config()
    test_index_prefix = "dummy/index_prefix"
    
    retriever = MemvidRetriever(index_file_path_prefix=test_index_prefix, config=test_config)
    
    MockDatabaseManager.assert_called_once_with(db_path=test_config["database"]["path"])
    MockIndexManager.assert_called_once_with(config=test_config)
    mock_index_instance.load.assert_called_once_with(test_index_prefix)
    assert retriever.db_manager is mock_db_instance
    assert retriever.index_manager is mock_index_instance
    assert retriever.index_file_path_prefix == test_index_prefix

@patch('memvid.retriever.IndexManager')
@patch('memvid.retriever.DatabaseManager')
def test_search(MockDatabaseManager, MockIndexManager):
    mock_index_instance = MockIndexManager.return_value
    # MockDatabaseManager.return_value is retriever.db_manager (mock_db_instance)
    
    test_config = create_test_config()
    retriever = MemvidRetriever(index_file_path_prefix="dummy/prefix", config=test_config)

    mock_index_instance.search.return_value = [(1, 0.1), (2, 0.2)]
    
    # Mock the internal method that gets data from DB and is cached
    # Use autospec=True to ensure the mock has the same signature as the real method
    with patch.object(retriever, '_get_db_chunk_details', autospec=True) as mock_get_details:
        # Define what _get_db_chunk_details returns for each call
        mock_get_details.side_effect = [
            {"chunk_id": 1, "text_content": "Text for chunk 1", "metadata": {"source": "A"}},
            {"chunk_id": 2, "text_content": "Text for chunk 2", "metadata": {"source": "B"}}
        ]

        results = retriever.search("test query", top_k=2)

        mock_index_instance.search.assert_called_once_with("test query", 2)
        # Check that _get_db_chunk_details was called with the IDs from index_manager.search
        mock_get_details.assert_has_calls([call(1), call(2)], any_order=False)
        assert results == ["Text for chunk 1", "Text for chunk 2"]

@patch('memvid.retriever.IndexManager')
@patch('memvid.retriever.DatabaseManager')
def test_search_with_metadata(MockDatabaseManager, MockIndexManager):
    mock_index_instance = MockIndexManager.return_value
    
    test_config = create_test_config()
    retriever = MemvidRetriever(index_file_path_prefix="dummy/prefix", config=test_config)

    mock_index_instance.search.return_value = [(1, 0.1)] # Search returns (id, distance)
    
    with patch.object(retriever, '_get_db_chunk_details', autospec=True) as mock_get_details:
        mock_get_details.return_value = {"chunk_id": 1, "text_content": "Text 1", "metadata": {"source": "docA"}}

        results = retriever.search_with_metadata("query", top_k=1)

        mock_index_instance.search.assert_called_once_with("query", 1)
        mock_get_details.assert_called_once_with(1)

        assert len(results) == 1
        assert results[0]["text"] == "Text 1"
        assert results[0]["chunk_id"] == 1
        assert results[0]["metadata"] == {"source": "docA"}
        assert "score" in results[0]
        # Check score calculation: 1.0 / (1.0 + distance)
        assert results[0]["score"] == pytest.approx(1.0 / (1.0 + 0.1))


@patch('memvid.retriever.IndexManager')
@patch('memvid.retriever.DatabaseManager')
def test_get_chunk_by_id(MockDatabaseManager, MockIndexManager):
    test_config = create_test_config()
    retriever = MemvidRetriever(index_file_path_prefix="dummy/prefix", config=test_config)

    with patch.object(retriever, '_get_db_chunk_details', autospec=True) as mock_get_details:
        # Test successful retrieval
        mock_get_details.return_value = {"chunk_id": 1, "text_content": "Specific text", "metadata": {}}
        text = retriever.get_chunk_by_id(1)
        assert text == "Specific text"
        mock_get_details.assert_called_once_with(1)

        # Test retrieval of a non-existent ID
        mock_get_details.reset_mock() # Reset call history for the next assertion
        mock_get_details.return_value = None # Simulate chunk not found
        text_none = retriever.get_chunk_by_id(999)
        assert text_none is None
        mock_get_details.assert_called_once_with(999)

@patch('memvid.retriever.IndexManager')
@patch('memvid.retriever.DatabaseManager')
def test_cache_operations(MockDatabaseManager, MockIndexManager):
    mock_db_instance = MockDatabaseManager.return_value

    test_config = create_test_config() # Uses retrieval.cache_size
    retriever = MemvidRetriever(index_file_path_prefix="dummy/prefix", config=test_config)
    
    # We mock the db_manager's method that _get_db_chunk_details calls INTERNALLY
    # This way, we test the caching behavior of _get_db_chunk_details itself
    mock_db_instance.get_chunk_by_id.return_value = {"chunk_id": 1, "text_content": "Cached text", "metadata": {}}

    # Call first time - should hit DB (via the mock)
    res1 = retriever._get_db_chunk_details(1)
    mock_db_instance.get_chunk_by_id.assert_called_once_with(1)
    assert res1["text_content"] == "Cached text"

    # Call second time - should use cache, so mock_db_instance.get_chunk_by_id is NOT called again
    res2 = retriever._get_db_chunk_details(1)
    mock_db_instance.get_chunk_by_id.assert_called_once() # Still called only once
    assert res2["text_content"] == "Cached text"

    # Clear cache
    retriever.clear_cache() # This calls _get_db_chunk_details.cache_clear()
    
    # Call again - should hit DB again (via the mock)
    res3 = retriever._get_db_chunk_details(1)
    assert mock_db_instance.get_chunk_by_id.call_count == 2 # Called again
    assert res3["text_content"] == "Cached text"

@patch('memvid.retriever.IndexManager')
@patch('memvid.retriever.DatabaseManager')
def test_retriever_stats(MockDatabaseManager, MockIndexManager):
    mock_index_instance = MockIndexManager.return_value
    mock_db_instance = MockDatabaseManager.return_value
    
    # Mock the db_path to be a Path-like object for str() conversion
    mock_db_path = Mock(spec=Path)
    mock_db_path.__str__.return_value = "dummy/db.sqlite"
    mock_db_instance.db_path = mock_db_path

    test_config = create_test_config()
    retriever = MemvidRetriever(index_file_path_prefix="dummy/prefix", config=test_config)

    mock_index_instance.get_stats.return_value = {"total_indexed_chunks": 10}
    
    # Simulate some cache activity for cache_info()
    # We need to ensure the real _get_db_chunk_details is called, so we mock the
    # underlying db_manager.get_chunk_by_id that it calls.
    with patch.object(retriever.db_manager, 'get_chunk_by_id') as mock_get_from_db:
        mock_get_from_db.return_value = {"text_content": "text", "chunk_id":1, "metadata":{}}
        retriever._get_db_chunk_details(1) # Call 1 (miss for lru, hit for db)
        retriever._get_db_chunk_details(1) # Call 2 (hit for lru)
        retriever._get_db_chunk_details(2) # Call 3 (miss for lru, hit for db)

    stats = retriever.get_stats()

    assert stats["index_file_prefix"] == "dummy/prefix"
    assert stats["db_path"] == "dummy/db.sqlite" # Check str conversion
    assert "cache_stats" in stats
    assert stats["index_stats"] == {"total_indexed_chunks": 10}

    cache_stats = stats["cache_stats"]
    assert "hits" in cache_stats
    assert "misses" in cache_stats
    assert cache_stats["currsize"] == 2 # Two unique items (1 and 2) added to cache
    # Exact hits/misses:
    # Call 1 (id 1): miss in lru, db call
    # Call 2 (id 1): hit in lru
    # Call 3 (id 2): miss in lru, db call
    # So, for _get_db_chunk_details cache: 1 hit, 2 misses.
    assert cache_stats["hits"] == 1
    assert cache_stats["misses"] == 2
    assert cache_stats["maxsize"] == test_config["retrieval"]["cache_size"]
