# tests/test_chat.py
import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

# Adjust import path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memvid.chat import MemvidChat, OPENAI_AVAILABLE # Import OPENAI_AVAILABLE for conditional mocking
from memvid.retriever import MemvidRetriever # For spec in mock
from memvid.config import get_default_config # For default config

TEST_INDEX_PREFIX = "dummy_index_prefix"

@pytest.fixture
def mock_retriever_instance():
    """Fixture for a mocked MemvidRetriever instance."""
    retriever = Mock(spec=MemvidRetriever)
    retriever.search.return_value = ["context chunk 1", "context chunk 2"]
    retriever.search_with_metadata.return_value = [
        {"text": "meta chunk 1", "score": 0.9, "chunk_id": 1, "metadata": {"s":1}},
        {"text": "meta chunk 2", "score": 0.8, "chunk_id": 2, "metadata": {"s":2}},
    ]
    # Mock get_stats if necessary for chat.get_stats()
    retriever.get_stats.return_value = {"db_path": "dummy_db", "cache_stats": {}, "index_stats": {}}
    return retriever

@pytest.fixture
def mock_openai_client_fixture(): # Renamed to avoid conflict with argument name
    """Fixture for a mocked OpenAI client, if OPENAI_AVAILABLE is True."""
    if not OPENAI_AVAILABLE:
        return None

    mock_client = MagicMock()
    mock_choice = Mock()
    mock_choice.message.content = "LLM response"
    mock_completion = Mock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client

# Patch paths for MemvidRetriever and OpenAI client within memvid.chat module
@patch('memvid.chat.MemvidRetriever')
@patch('memvid.chat.OpenAI', new_callable=lambda: MagicMock(name="MockOpenAIClientInTest") if OPENAI_AVAILABLE else None)
def test_chat_initialization(MockOpenAIClient, MockMemvidRetriever, mock_retriever_instance, mock_openai_client_fixture):
    MockMemvidRetriever.return_value = mock_retriever_instance
    default_config = get_default_config() # Get a default config to pass

    # Case 1: LLM available and API key provided
    if OPENAI_AVAILABLE and MockOpenAIClient is not None:
        MockOpenAIClient.return_value = mock_openai_client_fixture # Use the fixture's mock
        chat = MemvidChat(index_file_path_prefix=TEST_INDEX_PREFIX, llm_api_key="fake_key", config=default_config)
        MockMemvidRetriever.assert_called_once_with(index_file_path_prefix=TEST_INDEX_PREFIX, config=chat.config)
        MockOpenAIClient.assert_called_once_with(api_key="fake_key")
        assert chat.llm_client is not None
        assert chat.retriever is mock_retriever_instance

    # Reset mocks for the next case if OPENAI_AVAILABLE is True
    if OPENAI_AVAILABLE and MockOpenAIClient is not None:
        MockOpenAIClient.reset_mock()
    MockMemvidRetriever.reset_mock()

    # Case 2: LLM not available (simulated by no API key or OPENAI_AVAILABLE=False)
    # If OPENAI_AVAILABLE is true, but no key, OpenAI() is not called.
    # If OPENAI_AVAILABLE is false, MockOpenAIClient is None.
    if OPENAI_AVAILABLE and MockOpenAIClient is not None:
        # Test with OPENAI_AVAILABLE=True, but no API key
        chat_no_key = MemvidChat(index_file_path_prefix=TEST_INDEX_PREFIX, config=default_config) # No API key
        MockMemvidRetriever.assert_called_with(index_file_path_prefix=TEST_INDEX_PREFIX, config=chat_no_key.config)
        MockOpenAIClient.assert_not_called() # OpenAI should not be called if no key
        assert chat_no_key.llm_client is None
    else: # OPENAI_AVAILABLE is False
        chat_no_lib = MemvidChat(index_file_path_prefix=TEST_INDEX_PREFIX, config=default_config)
        MockMemvidRetriever.assert_called_with(index_file_path_prefix=TEST_INDEX_PREFIX, config=chat_no_lib.config)
        assert chat_no_lib.llm_client is None


@patch('memvid.chat.MemvidRetriever')
@patch('memvid.chat.OpenAI', new_callable=lambda: MagicMock(name="MockOpenAIClientInChat") if OPENAI_AVAILABLE else None)
def test_chat_method_with_llm(MockOpenAIClient, MockMemvidRetriever, mock_retriever_instance, mock_openai_client_fixture):
    if not OPENAI_AVAILABLE or MockOpenAIClient is None:
        pytest.skip("OpenAI library not available or mock setup failed, skipping LLM test.")

    MockMemvidRetriever.return_value = mock_retriever_instance
    MockOpenAIClient.return_value = mock_openai_client_fixture # Use the fixture

    chat = MemvidChat(index_file_path_prefix=TEST_INDEX_PREFIX, llm_api_key="fake_key")
    chat.start_session()
    response = chat.chat("User query")

    mock_retriever_instance.search.assert_called_once_with("User query", top_k=chat.context_chunks)
    mock_openai_client_fixture.chat.completions.create.assert_called_once()
    args, kwargs = mock_openai_client_fixture.chat.completions.create.call_args

    assert "User query" in kwargs["messages"][-1]["content"]
    assert "context chunk 1" in kwargs["messages"][-1]["content"]
    assert response == "LLM response"
    assert len(chat.conversation_history) == 2

@patch('memvid.chat.MemvidRetriever')
def test_chat_method_no_llm(MockMemvidRetriever, mock_retriever_instance):
    MockMemvidRetriever.return_value = mock_retriever_instance

    with patch('memvid.chat.OPENAI_AVAILABLE', False):
        chat = MemvidChat(index_file_path_prefix=TEST_INDEX_PREFIX)
        chat.start_session()
        response = chat.chat("User query")

        mock_retriever_instance.search.assert_called_once_with("User query", top_k=chat.context_chunks)
        assert "Based on the knowledge base" in response
        assert "context chunk 1" in response
        assert len(chat.conversation_history) == 2

@patch('memvid.chat.MemvidRetriever')
def test_search_context(MockMemvidRetriever, mock_retriever_instance):
    MockMemvidRetriever.return_value = mock_retriever_instance
    chat = MemvidChat(index_file_path_prefix=TEST_INDEX_PREFIX)

    results = chat.search_context("some query", top_k=3)
    mock_retriever_instance.search_with_metadata.assert_called_once_with("some query", top_k=3)
    assert len(results) == 2
    assert results[0]["text"] == "meta chunk 1"

@patch('memvid.chat.MemvidRetriever') # Patch retriever for __init__
def test_session_management(MockMemvidRetrieverForSession, tmp_path, mock_retriever_instance):
    # mock_retriever_instance isn't directly used by MemvidChat here, but good to have a consistent mock
    MockMemvidRetrieverForSession.return_value = mock_retriever_instance

    chat = MemvidChat(index_file_path_prefix=TEST_INDEX_PREFIX)

    chat.start_session("test_session_1")
    assert chat.session_id == "test_session_1"
    assert len(chat.conversation_history) == 0

    # Simulate chat messages being added (normally via chat.chat())
    chat.conversation_history.append({"role": "user", "content": "hi", "timestamp": "t1"})
    chat.conversation_history.append({"role": "assistant", "content": "hello", "timestamp": "t2"})

    history = chat.get_history()
    assert len(history) == 2
    assert history[0]["content"] == "hi"

    export_file = tmp_path / "session_export.json"
    chat.export_session(str(export_file))
    assert export_file.exists()
    with open(export_file, 'r') as f:
        data = json.load(f)
        assert data["session_id"] == "test_session_1"
        assert len(data["history"]) == 2

    chat.reset_session() # This also calls start_session implicitly if no session_id is active
    assert len(chat.conversation_history) == 0

    # Test load_session
    dummy_session_file = tmp_path / "dummy_load.json"
    dummy_data = {
        "session_id": "loaded_session",
        "history": [{"role":"user", "content":"loaded_msg", "timestamp":"t3"}]
        # Add other fields if load_session expects them, e.g. config, though current load_session doesn't use them
    }
    with open(dummy_session_file, 'w') as f:
        json.dump(dummy_data, f)

    chat.load_session(str(dummy_session_file))
    assert chat.session_id == "loaded_session"
    assert len(chat.conversation_history) == 1
    assert chat.conversation_history[0]["content"] == "loaded_msg"

@patch('memvid.chat.MemvidRetriever')
def test_get_stats(MockMemvidRetriever, mock_retriever_instance):
    MockMemvidRetriever.return_value = mock_retriever_instance

    with patch('memvid.chat.OPENAI_AVAILABLE', False): # Simulate no LLM
        chat = MemvidChat(index_file_path_prefix=TEST_INDEX_PREFIX)
        chat.start_session() # Sets session_id
        # Simulate one message pair for message_count
        chat.conversation_history.append({"role": "user", "content": "q", "timestamp":"t1"})
        chat.conversation_history.append({"role": "assistant", "content": "a", "timestamp":"t2"})


        stats = chat.get_stats()
        assert stats["session_id"] == chat.session_id
        assert stats["message_count"] == 2 # Counts both user and assistant messages
        assert stats["llm_available"] is False
        assert stats["retriever_stats"] == mock_retriever_instance.get_stats.return_value
        mock_retriever_instance.get_stats.assert_called_once() # Verify it was called
