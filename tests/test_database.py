# tests/test_database.py
import pytest
import sqlite3
import zlib
import json
from pathlib import Path
import os # For cleaning up test DB file

# Adjust import path if tests are run from root or a different working directory
# This assumes 'memvid' is in the python path (e.g. installed with pip install -e .)
# Or that PYTHONPATH is set correctly. For local testing:
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memvid.database import DatabaseManager # Assuming memvid/database.py

# Test database file name
TEST_DB_FILE = "test_memvid_memory.db"

@pytest.fixture
def db_manager(tmp_path):
    """Fixture to create and cleanup a DatabaseManager instance for testing."""
    # Use tmp_path from pytest for a temporary directory
    db_path = tmp_path / TEST_DB_FILE
    manager = DatabaseManager(db_path=str(db_path))
    yield manager
    # Cleanup: close connection if any and delete file
    # DatabaseManager doesn't hold a persistent connection, so just deleting is fine.
    if os.path.exists(db_path):
        try:
            # Attempt to allow GC to close file handles if any tests failed mid-operation
            del manager
            os.unlink(db_path)
        except Exception as e:
            print(f"Error during test DB cleanup: {e}")


def test_db_initialization(db_manager):
    """Test if the database and tables are created on initialization."""
    assert db_manager.db_path.exists()
    try:
        with sqlite3.connect(db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks';")
            assert cursor.fetchone() is not None, "'chunks' table should exist."
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='info';")
            assert cursor.fetchone() is not None, "'info' table should exist."
            cursor.execute("SELECT value FROM info WHERE key='schema_version';")
            assert cursor.fetchone()[0] == "1.0", "Schema version should be set."
    except sqlite3.Error as e:
        pytest.fail(f"Database connection or query failed: {e}")


def test_add_and_get_chunk(db_manager):
    """Test adding a chunk and retrieving it."""
    text_content = "This is a test chunk."
    metadata = {"source": "test_doc", "page": 1}

    chunk_id = db_manager.add_chunk(text_content, metadata)
    assert isinstance(chunk_id, int)
    assert chunk_id > 0

    retrieved_chunk = db_manager.get_chunk_by_id(chunk_id)
    assert retrieved_chunk is not None
    assert retrieved_chunk["chunk_id"] == chunk_id
    assert retrieved_chunk["text_content"] == text_content
    assert retrieved_chunk["metadata"] == metadata
    assert retrieved_chunk["original_length"] == len(text_content)

    # Verify compression by checking the raw compressed_content (optional, more involved)
    try:
        with sqlite3.connect(db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT compressed_content FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            assert row is not None
            compressed_db_content = row[0]
            assert zlib.decompress(compressed_db_content).decode('utf-8') == text_content
    except sqlite3.Error as e:
        pytest.fail(f"Database query for compressed content failed: {e}")


def test_add_chunk_empty_content(db_manager):
    """Test adding a chunk with empty content raises ValueError."""
    with pytest.raises(ValueError, match="Text content cannot be empty."):
        db_manager.add_chunk("", {"source": "empty_test"})


def test_get_non_existent_chunk(db_manager):
    """Test retrieving a non-existent chunk returns None."""
    assert db_manager.get_chunk_by_id(99999) is None


def test_get_all_chunks_for_indexing(db_manager):
    """Test retrieving all chunks for indexing."""
    chunks_data = [
        ("Chunk 1 text", {"id": "c1"}),
        ("Second chunk of text", {"id": "c2"}),
        ("The third and final chunk", {"id": "c3"}),
    ]
    expected_chunks = []
    for i, (text, meta) in enumerate(chunks_data):
        chunk_id = db_manager.add_chunk(text, meta)
        expected_chunks.append((chunk_id, text)) # Store as (id, text)

    all_chunks = db_manager.get_all_chunks_for_indexing()
    assert len(all_chunks) == len(chunks_data)

    # Sort both by chunk_id before comparing if order isn't guaranteed (it is by query)
    # all_chunks is already ordered by chunk_id from the DB query.
    assert all_chunks == expected_chunks


def test_get_total_chunks(db_manager):
    """Test getting the total number of chunks."""
    assert db_manager.get_total_chunks() == 0
    db_manager.add_chunk("Chunk A", {})
    assert db_manager.get_total_chunks() == 1
    db_manager.add_chunk("Chunk B", {})
    assert db_manager.get_total_chunks() == 2


def test_clear_all_chunks(db_manager):
    """Test clearing all chunks from the database."""
    db_manager.add_chunk("Content to be cleared", {})
    db_manager.add_chunk("More content", {})
    assert db_manager.get_total_chunks() == 2

    db_manager.clear_all_chunks()
    assert db_manager.get_total_chunks() == 0

    # Test if autoincrement is reset (IDs should start from 1 again)
    new_chunk_id = db_manager.add_chunk("New chunk after clear", {})
    assert new_chunk_id == 1


def test_info_table_operations(db_manager):
    """Test set_info and get_info methods."""
    assert db_manager.get_info("test_key") is None # Should not exist initially

    db_manager.set_info("test_key", "test_value")
    assert db_manager.get_info("test_key") == "test_value"

    db_manager.set_info("test_key_int", 123)
    assert db_manager.get_info("test_key_int") == 123

    complex_data = {"a": 1, "b": [2, 3], "c": "hello"}
    db_manager.set_info("test_key_complex", complex_data)
    assert db_manager.get_info("test_key_complex") == complex_data

    # Test overwriting
    db_manager.set_info("test_key", "new_value")
    assert db_manager.get_info("test_key") == "new_value"

def test_add_chunk_no_metadata(db_manager):
    text_content = "Chunk without explicit metadata."
    chunk_id = db_manager.add_chunk(text_content) # Call with metadata=None (default)
    assert chunk_id > 0
    retrieved = db_manager.get_chunk_by_id(chunk_id)
    assert retrieved is not None
    assert retrieved["text_content"] == text_content
    assert retrieved["metadata"] == {} # Should default to empty dict

def test_db_path_creation(tmp_path):
    """Test that DatabaseManager creates the db file at the specified path."""
    db_dir = tmp_path / "custom_db_dir"
    db_file = db_dir / "custom_name.db"
    # db_dir does not exist yet

    assert not db_dir.exists()
    manager = DatabaseManager(db_path=str(db_file))
    assert db_dir.exists() # Directory should be created
    assert db_file.exists() # DB file should be created
    del manager # Allow cleanup
    if os.path.exists(db_file):
        os.unlink(db_file)
    if os.path.exists(db_dir):
        # Only remove if empty, though it should be after db file removal
        if not any(os.scandir(db_dir)):
            os.rmdir(db_dir)
