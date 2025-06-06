# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Memvid is a Python library for QR code video-based AI memory that enables:
- Chunking and encoding text data into QR code videos
- Fast semantic search and retrieval from QR videos
- Conversational AI interface with context-aware memory
- Web-based Streamlit interface for easy interaction
- Google Colab notebook for cloud-based usage

## Key Architecture

### Core Components
- **MemvidEncoder** (memvid/encoder.py): Handles text chunking and QR video creation
- **MemvidRetriever** (memvid/retriever.py): Fast semantic search, QR frame extraction, context assembly
- **MemvidChat** (memvid/chat.py): Manages conversations, context retrieval, and LLM interface
- **IndexManager** (memvid/index.py): Embedding generation, storage, and vector search
- **Streamlit Interface** (streamlit_chat.py): Web-based GUI for memory creation and chat
- **Colab Notebook** (Memvid_colab.ipynb): Cloud-based environment for easy testing

### Data Flow
1. Text chunks → Embeddings → QR codes → Video frames
2. Query → Semantic search → Frame extraction → QR decode → Context
3. Context + History → LLM → Response

## Development Commands

```bash
# Create and activate virtual environment
python -m venv .memvid
source .memvid/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run specific test
pytest tests/test_encoder.py::TestSpecificFunction

# Install package in development mode
pip install -e .

# Run Streamlit web interface
streamlit run streamlit_chat.py
```

## Key Dependencies
- qrcode, Pillow: QR generation
- opencv-python: Video processing
- pyzbar: QR decoding
- sentence-transformers: Semantic embeddings
- numpy: Vector operations
- openai: LLM integration (default: GPT-4.1)
- streamlit: Web interface
- google-generativeai: Gemini support
- anthropic: Claude support

## Performance Requirements
- Retrieval (search + QR decode) must be < 2 seconds for 1M chunks
- Use batching and parallel processing for frame extraction
- Implement caching for hot frames and common queries

## Implementation Notes
- Vector DB options: FAISS, Annoy, or Chroma for scalability
- LLM backend is pluggable (OpenAI GPT-4.1 default, Claude, Gemini)
- Thread/process pools for parallel QR decoding
- Disk-based index for large-scale deployments
- Streamlit interface provides user-friendly memory management
- Colab notebook enables cloud-based testing without local setup