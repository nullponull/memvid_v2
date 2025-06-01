"""
Interactive chat interface for Memvid
"""

import os
import time
from typing import Optional, Dict, Any
from .chat import MemvidChat


def chat_with_memory(
    index_file_path_prefix: str, # Changed
    api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    show_stats: bool = True,
    export_on_exit: bool = True,
    session_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Start an interactive chat session with a memory (database + index).
    
    Args:
        index_file_path_prefix: Path prefix for index files (e.g., 'output/my_memory').
        api_key: OpenAI API key (or set OPENAI_API_KEY env var).
        llm_model: LLM model to use (default: gpt-3.5-turbo).
        show_stats: Show memory stats on startup.
        export_on_exit: Auto-export conversation on exit.
        session_dir: Directory to save session files (default: "output").
        config: Optional configuration.
        
    Commands:
        - 'search <query>': Show raw search results
        - 'stats': Show system statistics
        - 'export': Save conversation
        - 'clear': Clear conversation history
        - 'help': Show commands
        - 'exit' or 'quit': End session
        
    Example:
        >>> from memvid import chat_with_memory
        >>> # Ensure 'output/my_memory.faiss' and 'output/my_memory.indexinfo.json' exist
        >>> # and the database (e.g., 'memvid_memory.db') is populated.
        >>> chat_with_memory("output/my_memory")
    """
    # Set tokenizers parallelism to avoid warning
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Set default session directory
    if session_dir is None:
        session_dir = "output"
    os.makedirs(session_dir, exist_ok=True)
    
    # Check if index files exist
    faiss_file_to_check = f"{index_file_path_prefix}.faiss"
    indexinfo_file_to_check = f"{index_file_path_prefix}.indexinfo.json"

    if not os.path.exists(faiss_file_to_check):
        print(f"Error: Index FAISS file not found: {faiss_file_to_check}")
        print(f"Ensure you have run the memory building process and that '{faiss_file_to_check}' and preferably '{indexinfo_file_to_check}' exist.")
        return
    if not os.path.exists(indexinfo_file_to_check):
        print(f"Warning: Index info file not found: {indexinfo_file_to_check}. Index might load with default parameters if configuration changed.")

    # Initialize chat
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    print("Initializing Memvid Chat...")
    chat = MemvidChat(index_file_path_prefix, llm_api_key=api_key, llm_model=llm_model, config=config)
    chat.start_session()
    
    # Show stats if requested
    if show_stats:
        stats = chat.get_stats()
        ret_stats = stats.get('retriever_stats', {})
        idx_stats = ret_stats.get('index_stats', {})
        print(f"\nMemory loaded: {idx_stats.get('total_indexed_chunks', 'N/A')} chunks indexed.")
        print(f"  Index path prefix: {ret_stats.get('index_file_prefix', 'N/A')}")
        print(f"  Database path: {ret_stats.get('db_path', 'N/A')}")
        if stats.get('llm_available'):
            print(f"LLM: {stats.get('llm_model', 'N/A')}")
        else:
            print("LLM: Not available (context-only mode)")
    
    print("\nType 'help' for commands, 'exit' to quit")
    print("-" * 50)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            lower_input = user_input.lower()
            
            if lower_input in ['exit', 'quit', 'q']:
                break
                
            elif lower_input == 'help':
                print("\nCommands:")
                print("  search <query> - Show raw search results")
                print("  stats         - Show system statistics")
                print("  export        - Save conversation")
                print("  clear         - Clear conversation history")
                print("  help          - Show this help")
                print("  exit/quit     - End session")
                continue
                
            elif lower_input == 'stats':
                stats = chat.get_stats()
                ret_stats = stats.get('retriever_stats', {})
                idx_s = ret_stats.get('index_stats', {})
                cache_s = ret_stats.get('cache_stats', {})

                print(f"\nSession Messages: {stats.get('message_count', 'N/A')}")
                print(f"LLM Model: {stats.get('llm_model', 'N/A')} ({'Available' if stats.get('llm_available') else 'Not Available'})")
                print(f"Retriever:")
                print(f"  Index Path Prefix: {ret_stats.get('index_file_prefix', 'N/A')}")
                print(f"  DB Path: {ret_stats.get('db_path', 'N/A')}")
                if cache_s.get('info') != 'Cache info not available for _get_db_chunk_details.':
                     print(f"  Cache: {cache_s.get('currsize',0)}/{cache_s.get('maxsize',0)} (Hits:{cache_s.get('hits',0)}, Misses:{cache_s.get('misses',0)})")
                else:
                    print(f"  Cache: {cache_s.get('info', 'N/A')}")
                print(f"  Index Stats:")
                for k, v in idx_s.items():
                    print(f"    {k}: {v}")
                continue
                
            elif lower_input == 'export':
                export_file = os.path.join(session_dir, f"memvid_session_{chat.session_id}.json")
                chat.export_session(export_file)
                print(f"Exported to: {export_file}")
                continue
                
            elif lower_input == 'clear':
                chat.reset_session()
                chat.start_session()
                print("Conversation cleared.")
                continue
                
            elif lower_input.startswith('search '):
                query = user_input[7:]
                print(f"\nSearching: '{query}'")
                start_time = time.time()
                results = chat.search_context(query, top_k=5)
                elapsed = time.time() - start_time
                print(f"Found {len(results)} results in {elapsed:.3f}s:\n")
                for i, result in enumerate(results[:3]):
                    print(f"{i+1}. [Score: {result['score']:.3f}] {result['text'][:100]}...")
                continue
            
            # Regular chat
            print("\nAssistant: ", end="", flush=True)
            start_time = time.time()
            response = chat.chat(user_input)
            elapsed = time.time() - start_time
            
            print(response)
            print(f"\n[{elapsed:.1f}s]", end="")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    # Export on exit if requested
    if export_on_exit and chat.get_history():
        export_file = os.path.join(session_dir, f"memvid_session_{chat.session_id}.json")
        chat.export_session(export_file)
        print(f"\nSession saved to: {export_file}")
    
    print("Goodbye!")


def quick_chat(index_file_path_prefix: str, query: str, api_key: Optional[str] = None) -> str:
    """
    Quick one-off query without interactive loop.
    
    Args:
        index_file_path_prefix: Path prefix for index files.
        query: Question to ask.
        api_key: OpenAI API key (optional).
        
    Returns:
        Response string.
        
    Example:
        >>> from memvid import quick_chat
        >>> # Ensure index and DB are set up
        >>> response = quick_chat("output/my_memory", "What is quantum computing?")
        >>> print(response)
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Add basic file check for quick_chat as well for better UX
    faiss_file_to_check = f"{index_file_path_prefix}.faiss"
    if not os.path.exists(faiss_file_to_check):
        return f"Error: Index FAISS file not found: {faiss_file_to_check}. Cannot proceed with quick_chat."

    chat = MemvidChat(index_file_path_prefix, llm_api_key=api_key)
    return chat.chat(query)