#!/usr/bin/env python3
"""
Example: Interactive conversation using MemvidChat
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import MemvidChat
import time


def print_search_results(results):
    """Pretty print search results"""
    print("\nRelevant context found:")
    print("-" * 50)
    for i, result in enumerate(results[:3]): # Show top 3 results
        print(f"\n[{i+1}] Score: {result.get('score', 0.0):.3f}")
        print(f"Text: {result.get('text', '')[:150]}...")
        print(f"Chunk ID: {result.get('chunk_id', 'N/A')}")
        custom_meta = result.get('metadata', {})
        if custom_meta:
            print(f"Metadata: {custom_meta}")


def main():
    print("Memvid Example: Interactive Chat with Memory")
    print("=" * 50)
    
    # Define path prefix for index files
    index_file_path_prefix = "output/memory_index"

    # Check if memory files exist
    faiss_file_expected = f"{index_file_path_prefix}.faiss"
    info_file_expected = f"{index_file_path_prefix}.indexinfo.json" # Also check for this
    
    if not os.path.exists(faiss_file_expected) or not os.path.exists(info_file_expected):
        print("\nError: Memory files not found (expected .faiss and .indexinfo.json).")
        print(f"Searched for: {faiss_file_expected} and {info_file_expected}")
        print("Please run 'python examples/build_memory.py' first to create the memory.")
        return
    
    # Initialize chat
    print(f"\nLoading memory from index prefix: {index_file_path_prefix}")
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nNote: No OpenAI API key found (checked OPENAI_API_KEY env var). Chat will work in context-only mode.")
        # The MemvidChat class itself will also log a warning if API key is missing for OpenAI.
    
    chat = MemvidChat(index_file_path_prefix, llm_api_key=api_key)
    chat.start_session()
    
    # Get stats
    stats = chat.get_stats()
    ret_stats = stats.get('retriever_stats', {})
    idx_stats = ret_stats.get('index_stats', {})
    print(f"\nMemory loaded successfully!")
    print(f"  Total indexed chunks: {idx_stats.get('total_indexed_chunks', 'N/A')}")
    print(f"  LLM available: {stats.get('llm_available', False)}")
    if stats.get('llm_available'):
        print(f"  LLM model: {stats.get('llm_model', 'N/A')}")
    
    print("\nInstructions:")
    print("- Type your questions to search the memory")
    print("- Type 'search <query>' to see raw search results")
    print("- Type 'stats' to see system statistics")
    print("- Type 'export' to save conversation")
    print("- Type 'exit' or 'quit' to end the session")
    print("-" * 50)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
                
            elif user_input.lower() == 'stats':
                stats = chat.get_stats()
                ret_stats = stats.get('retriever_stats', {})
                idx_s = ret_stats.get('index_stats', {})
                cache_s = ret_stats.get('cache_stats', {})

                print("\nSystem Statistics:")
                print(f"  Session Messages: {stats.get('message_count', 'N/A')}")
                print(f"  LLM Model: {stats.get('llm_model', 'N/A')} ({'Available' if stats.get('llm_available') else 'Not Available'})")
                print(f"  Retriever:")
                print(f"    Index Path Prefix: {ret_stats.get('index_file_prefix', 'N/A')}")
                print(f"    Database path: {ret_stats.get('db_path', 'N/A')}")
                if cache_s and cache_s.get('info') != 'Cache info not available for _get_db_chunk_details.': # Check if cache_s is not None
                     print(f"    Cache: {cache_s.get('currsize',0)}/{cache_s.get('maxsize',0)} (Hits:{cache_s.get('hits',0)}, Misses:{cache_s.get('misses',0)})")
                else:
                    print(f"    Cache: Info not available or N/A")
                print(f"    Index Stats:")
                for k, v in idx_s.items(): # Iterate over idx_s which is already a dict
                    print(f"      {k}: {v}")
                continue
                
            elif user_input.lower() == 'export':
                export_file = f"output/session_{chat.session_id}.json"
                chat.export_session(export_file)
                print(f"Session exported to: {export_file}")
                continue
                
            elif user_input.lower().startswith('search '):
                query = user_input[7:]
                print(f"\nSearching for: '{query}'")
                start_time = time.time()
                results = chat.search_context(query, top_k=5)
                elapsed = time.time() - start_time
                print(f"Search completed in {elapsed:.3f} seconds")
                print_search_results(results)
                continue
            
            # Regular chat
            print("\nAssistant: ", end="", flush=True)
            start_time = time.time()
            response = chat.chat(user_input)
            elapsed = time.time() - start_time
            
            print(response)
            print(f"\n[Response time: {elapsed:.2f}s]")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    # Export session on exit
    if chat.get_history():
        export_file = f"output/session_{chat.session_id}.json"
        chat.export_session(export_file)
        print(f"\nSession saved to: {export_file}")


if __name__ == "__main__":
    main()