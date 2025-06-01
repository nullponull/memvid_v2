#!/usr/bin/env python3
"""
Book memory example using chat_with_memory
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


from memvid import MemvidEncoder, chat_with_memory

# Book PDF path - Memvid will handle PDF parsing automatically
book_pdf = "data/bitcoin.pdf"  # Replace with your PDF path

# Build memory (DB + FAISS index) from PDF
# video_path is no longer used for the core memory.
index_file_path_prefix = "output/book_memory_index" # Prefix for .faiss and .indexinfo.json

# Create output directory with subdirectory for sessions
# This output dir is for chat sessions. The main DB and index files will go into "output/"
# based on the index_file_path_prefix.
os.makedirs("output/book_chat", exist_ok=True)

# Encode PDF to database and build FAISS index
encoder = MemvidEncoder() # DB will be created/used based on default config (e.g. ./memvid_memory.db)
print(f"Adding PDF: {book_pdf} to memory...")
encoder.add_pdf(book_pdf) # PDF content is chunked and added to the DB
print(f"Building memory files with prefix: {index_file_path_prefix}...")
encoder.build_memory(index_file_path_prefix) # Creates FAISS index from DB content
print(f"Created book memory (DB and index files with prefix: {index_file_path_prefix})")

# Get API key from environment or use your own
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\nNote: Set OPENAI_API_KEY environment variable for full LLM responses.")
    print("Without it, you'll only see raw context chunks or LLM-less responses.\n")

# Chat with the book - interactive session
print("\n📚 Chat with your book! Ask questions about the content.")
print("Example questions:")
print("- 'What is this document about?'")
print("- 'What are the key concepts explained?'\n")

# Call updated chat_with_memory
chat_with_memory(index_file_path_prefix, api_key=api_key, session_dir="output/book_chat")