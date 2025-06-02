import pytest
from unittest.mock import Mock, patch, call
import os
from pathlib import Path
import tempfile

from memvid.encoder import MemvidEncoder
# Assuming memvid.config.get_default_config is available for default config values
from memvid.config import get_default_config

# Helper to create a default-like config for tests
def create_test_config():
    config = get_default_config()
    # Override DB path for predictability in tests if needed, though it's mocked.
    config["database"]["path"] = "dummy/test_encoder_memory.db"
    return config

@patch('memvid.encoder.IndexManager')
@patch('memvid.encoder.DatabaseManager')
def test_encoder_initialization(MockDatabaseManager, MockIndexManager):
    """Test encoder initialization with mocked DB and Index managers."""
    mock_db_instance = MockDatabaseManager.return_value
    mock_index_instance = MockIndexManager.return_value

    test_config = create_test_config()
    encoder = MemvidEncoder(config=test_config)

    MockDatabaseManager.assert_called_once_with(db_path=test_config["database"]["path"])
    MockIndexManager.assert_called_once_with(config=test_config)
    assert encoder.db_manager is mock_db_instance
    assert encoder.index_manager is mock_index_instance

@patch('memvid.encoder.IndexManager')
@patch('memvid.encoder.DatabaseManager')
def test_add_chunks(MockDatabaseManager, MockIndexManager):
    """Test adding chunks calls db_manager.add_chunk for each chunk."""
    mock_db_instance = MockDatabaseManager.return_value
    test_config = create_test_config()
    encoder = MemvidEncoder(config=test_config)

    test_chunks = ["chunk1 text", "chunk2 text"]
    test_metadata_list = [{"source": "doc1"}, {"source": "doc2"}]

    encoder.add_chunks(test_chunks, metadata_list=test_metadata_list)
    
    expected_calls = [
        call(test_chunks[0], test_metadata_list[0]),
        call(test_chunks[1], test_metadata_list[1])
    ]
    mock_db_instance.add_chunk.assert_has_calls(expected_calls, any_order=False)

@patch('memvid.encoder.chunk_text') # Mock the utility function
@patch('memvid.encoder.IndexManager')
@patch('memvid.encoder.DatabaseManager')
def test_add_text(MockDatabaseManager, MockIndexManager, mock_chunk_text_util):
    """Test add_text chunks text and then calls db_manager.add_chunk."""
    mock_db_instance = MockDatabaseManager.return_value
    test_config = create_test_config()
    encoder = MemvidEncoder(config=test_config)

    sample_text = "This is a long text to be chunked."
    expected_chunked_texts = ["This is a long text", "to be chunked."]
    mock_chunk_text_util.return_value = expected_chunked_texts

    test_metadata = {"source": "test_document"}
    # Create a copy for assertion because it's modified in add_text by .copy()
    # The actual dict passed to add_chunk will be a copy of test_metadata.
    # So we assert that each call received a dict equal to test_metadata.

    encoder.add_text(sample_text, chunk_size=20, overlap=5, metadata=test_metadata)

    mock_chunk_text_util.assert_called_once_with(sample_text, 20, 5)

    # Check that db_manager.add_chunk was called correctly for each chunk
    # Each call to add_chunk receives its own copy of the metadata dictionary
    # (or an empty one if none provided).
    # So we check the content of the dictionary for each call.
    args_list = mock_db_instance.add_chunk.call_args_list
    assert len(args_list) == len(expected_chunked_texts)
    for i, expected_text_chunk in enumerate(expected_chunked_texts):
        actual_call_args = args_list[i][0] # Get positional args of the call
        assert actual_call_args[0] == expected_text_chunk
        assert actual_call_args[1] == test_metadata # check dictionary equality

@patch('memvid.encoder.IndexManager')
@patch('memvid.encoder.DatabaseManager')
def test_build_memory(MockDatabaseManager, MockIndexManager):
    """Test build_memory orchestrates IndexManager calls and returns correct stats."""
    mock_db_instance = MockDatabaseManager.return_value
    mock_index_instance = MockIndexManager.return_value
    
    # Setup mock return values
    mock_db_instance.get_total_chunks.return_value = 10
    # Mock Path object for db_path and its methods used in build_memory
    mock_db_path = Mock(spec=Path)
    mock_db_path.exists.return_value = True
    mock_db_path.stat.return_value = Mock(st_size=1024*1024*2) # 2MB
    mock_db_path.__str__.return_value = "dummy/test.db" # How it's converted for stats
    mock_db_instance.db_path = mock_db_path

    mock_index_instance.get_stats.return_value = {"indexed_count": 10, "model": "test_model"}

    test_config = create_test_config()
    encoder = MemvidEncoder(config=test_config)

    index_prefix = "output/test_memory_idx"

    # Ensure build_index_from_db exists on the mock
    mock_index_instance.build_index_from_db = Mock()

    build_stats = encoder.build_memory(index_prefix, show_progress=False)

    mock_index_instance.build_index_from_db.assert_called_once_with(mock_db_instance, show_progress=False)
    mock_index_instance.save.assert_called_once_with(index_prefix)

    assert build_stats["total_chunks_in_db"] == 10
    assert build_stats["index_file_prefix"] == index_prefix
    assert build_stats["index_stats"] == {"indexed_count": 10, "model": "test_model"}
    assert build_stats["database_stats"]["db_file"] == "dummy/test.db"
    assert build_stats["database_stats"]["db_size_mb"] == 2.0

@patch('memvid.encoder.IndexManager')
@patch('memvid.encoder.DatabaseManager')
def test_get_stats(MockDatabaseManager, MockIndexManager):
    """Test get_stats retrieves info from db_manager."""
    mock_db_instance = MockDatabaseManager.return_value
    test_config = create_test_config()
    encoder = MemvidEncoder(config=test_config)

    mock_db_instance.get_total_chunks.return_value = 5
    mock_db_path = Mock(spec=Path)
    mock_db_path.__str__.return_value = "dummy/test.db"
    mock_db_instance.db_path = mock_db_path
    
    stats = encoder.get_stats()

    assert stats["total_chunks_in_db"] == 5
    assert stats["db_path"] == "dummy/test.db"
    assert stats["config"] == test_config

@patch('memvid.encoder.IndexManager')
@patch('memvid.encoder.DatabaseManager')
def test_clear(MockDatabaseManager, MockIndexManager):
    """Test clear calls clear_all_chunks on db_manager and clear_index on index_manager."""
    mock_db_instance = MockDatabaseManager.return_value
    mock_index_instance = MockIndexManager.return_value

    # Ensure clear_index exists on the mock
    mock_index_instance.clear_index = Mock()

    test_config = create_test_config()
    encoder = MemvidEncoder(config=test_config)
    
    encoder.clear()

    mock_db_instance.clear_all_chunks.assert_called_once()
    mock_index_instance.clear_index.assert_called_once()

# Placeholder for PDF test - This is a more complex test to get right with mocking PyPDF2
# For brevity and focus, the full PyPDF2 mocking is simplified.
# In a real scenario, ensure PdfReader and page objects are mocked accurately.
@patch('memvid.encoder.PyPDF2') # Mock the entire PyPDF2 module at the point of import
@patch('memvid.encoder.chunk_text')
@patch('memvid.encoder.IndexManager')
@patch('memvid.encoder.DatabaseManager')
def test_add_pdf_basic(MockDatabaseManager, MockIndexManager, mock_chunk_text_util, MockPyPDF2):
    """Basic test for add_pdf to ensure it tries to process and add chunks."""
    mock_db_instance = MockDatabaseManager.return_value

    # Configure the mock PdfReader from the mocked PyPDF2 module
    mock_pdf_reader_instance = MockPyPDF2.PdfReader.return_value
    mock_page = Mock()
    mock_page.extract_text.return_value = "PDF page text."
    mock_pdf_reader_instance.pages = [mock_page]

    mock_chunk_text_util.return_value = ["PDF page text."]

    test_config = create_test_config()
    encoder = MemvidEncoder(config=test_config)

    # Mock Path.exists for the PDF file check within add_pdf
    with patch('memvid.encoder.Path') as mock_path_constructor:
        mock_path_instance = mock_path_constructor.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.name = "dummy.pdf" # For metadata

        # Mock open as well since add_pdf opens the file
        with patch('builtins.open', new_callable=Mock) as mock_open_file:
            encoder.add_pdf("dummy.pdf", chunk_size=100, overlap=10)

    MockPyPDF2.PdfReader.assert_called_once()
    mock_chunk_text_util.assert_called_once_with("PDF page text.\n\n", 100, 10)

    expected_metadata = {"source_pdf": "dummy.pdf", "pdf_total_pages": 1}
    # Check call to db_manager.add_chunk (similar to test_add_text)
    args_list = mock_db_instance.add_chunk.call_args_list
    assert len(args_list) == 1
    actual_call_args = args_list[0][0]
    assert actual_call_args[0] == "PDF page text."
    assert actual_call_args[1] == expected_metadata
