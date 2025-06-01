#!/usr/bin/env python3
"""
Simplified interactive chat example
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import chat_with_memory, quick_chat

def main():
    print("Memvid Simple Chat Examples")
    print("=" * 50)
    
    # video_file is no longer used for retriever/chat initialization
    index_file_path_prefix = "output/memory_index" # Path prefix for .faiss and .indexinfo.json
    
    # Check if memory (index files) exists
    faiss_file_expected = f"{index_file_path_prefix}.faiss"
    if not os.path.exists(faiss_file_expected):
        print(f"\nError: Index file {faiss_file_expected} not found.")
        print("Ensure you have run 'python examples/build_memory.py' first to create the memory files.")
        return
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") # Prefer environment variable
    if not api_key:
        # Try to fall back to a hardcoded one if you uncomment below, or just warn.
        # api_key = "your-api-key-here" # Replace or remove
        print("\nNote: Set OPENAI_API_KEY environment variable for full LLM responses.")
        print("Without it, you'll only see raw context chunks or LLM-less responses.\n")
        if not api_key: # If still not set after trying hardcoded (if any)
             print("Continuing without an API key specified for LLM.")
    
    print("\n1. Quick one-off query:")
    print("-" * 30)
    # Pass index_file_path_prefix instead of video_file and index_file
    response = quick_chat(index_file_path_prefix, "How many qubits did the quantum computer achieve?", api_key=api_key)
    print(f"Response: {response}")
    
    print("\n\n2. Interactive chat session:")
    print("-" * 30)
    print("Starting interactive mode...\n")
    
    # Pass index_file_path_prefix
    chat_with_memory(index_file_path_prefix, api_key=api_key)

if __name__ == "__main__":
    main()