#!/usr/bin/env python3
"""
Example: Create video memory and index from text data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import MemvidEncoder
import time


def main():
    # Example data - could be from files, databases, etc.
    chunks = [
        "The quantum computer achieved 100 qubits of processing power in March 2024.",
        "Machine learning models can now process over 1 trillion parameters efficiently.",
        "The new GPU architecture delivers 5x performance improvement for AI workloads.",
        "Cloud storage costs have decreased by 80% over the past five years.",
        "Quantum encryption methods are becoming standard for secure communications.",
        "Edge computing reduces latency to under 1ms for critical applications.",
        "Neural networks can now generate photorealistic images in real-time.",
        "Blockchain technology processes over 100,000 transactions per second.",
        "5G networks provide speeds up to 10 Gbps in urban areas.",
        "Autonomous vehicles have logged over 50 million miles of testing.",
        "Natural language processing accuracy has reached 98% for major languages.",
        "Robotic process automation saves companies millions in operational costs.",
        "Augmented reality glasses now have 8-hour battery life.",
        "Biometric authentication systems have false positive rates below 0.001%.",
        "Distributed computing networks utilize idle resources from millions of devices.",
        "Green data centers run entirely on renewable energy sources.",
        "AI assistants can understand context across multiple conversation turns.",
        "Cybersecurity AI detects threats 50x faster than traditional methods.",
        "Digital twins simulate entire cities for urban planning.",
        "Voice cloning technology requires only 3 seconds of audio sample.",
    ]
    
    print("Memvid Example: Building Memory (Database + Index)")
    print("=" * 50)
    
    # Create encoder
    # Configuration for database path can be implicitly handled by MemvidEncoder
    # using defaults from config.py, or explicitly passed if needed.
    # For this example, we'll rely on the default (memvid_memory.db in the current dir or as per config).
    encoder = MemvidEncoder()
    
    # Add chunks
    print(f"\nAdding {len(chunks)} chunks to encoder...")
    encoder.add_chunks(chunks)
    
    # You can also add from text with automatic chunking
    additional_text = """
    The future of computing lies in the convergence of multiple technologies.
    Quantum computing will solve problems that are intractable for classical computers.
    AI and machine learning will become embedded in every application.
    The edge and cloud will work together seamlessly to process data where it makes most sense.
    Privacy-preserving technologies will enable collaboration without exposing sensitive data.
    """
    
    print("\nAdding additional text with automatic chunking...")
    encoder.add_text(additional_text, chunk_size=100, overlap=20)
    
    # Get stats
    stats = encoder.get_stats()
    print(f"\nEncoder stats (after adding initial chunks):")
    print(f"  Total chunks in DB: {stats.get('total_chunks_in_db', 'N/A')}")
    print(f"  Database path: {stats.get('db_path', 'N/A')}")
    
    # Build memory (FAISS index + metadata json stored alongside, DB is separate)
    output_dir = "output" # Directory for index files
    os.makedirs(output_dir, exist_ok=True)
    
    # index_prefix is used for e.g. memory_index.faiss, memory_index.indexinfo.json
    index_prefix = os.path.join(output_dir, "memory_index")
    
    print(f"\nBuilding memory with index prefix: {index_prefix}")
    # The database itself (e.g., memvid_memory.db) will be created/updated based on encoder's config.
    
    start_time = time.time()
    build_stats = encoder.build_memory(index_prefix, show_progress=True)
    elapsed = time.time() - start_time
    
    print(f"\nBuild completed in {elapsed:.2f} seconds")

    db_s = build_stats.get("database_stats", {})
    print(f"\nDatabase stats:")
    print(f"  DB file: {db_s.get('db_file', 'N/A')}")
    print(f"  DB size: {db_s.get('db_size_mb', 0):.2f} MB")
    # 'total_chunks_in_db' is now the primary count from build_stats
    print(f"  Total chunks in DB (from build_stats): {build_stats.get('total_chunks_in_db', 'N/A')}")
    
    print("\nIndex stats:") # index_stats from build_stats directly
    idx_s = build_stats.get('index_stats', {})
    for key, value in idx_s.items():
        print(f"  {key}: {value}")
    
    print("\nSuccess! Memory created.")
    print(f"  Index files prefix: {build_stats.get('index_file_prefix')}")
    print(f"  Database file: {db_s.get('db_file')}") # From database_stats
    print(f"\nYou can now use this memory with chat examples (after they are updated for the new retriever).")


if __name__ == "__main__":
    main()