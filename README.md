# Memvid - Video-Based AI Memory 🧠📹

**The lightweight, game-changing solution for AI memory at scale**

[![PyPI version](https://badge.fury.io/py/memvid.svg)](https://pypi.org/project/memvid/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Memvid revolutionizes AI memory management by encoding text data into videos, enabling **lightning-fast semantic search** across millions of text chunks with **sub-second retrieval times**. Unlike traditional vector databases that consume massive amounts of RAM and storage, Memvid compresses your knowledge base into compact video files while maintaining instant access to any piece of information.

## 🎥 Demo

https://github.com/user-attachments/assets/ec550e93-e9c4-459f-a8a1-46e122b5851e



## ✨ Key Features

- 🎥 **Video-as-Database**: Store millions of text chunks in a single MP4 file
- 🔍 **Semantic Search**: Find relevant content using natural language queries
- 💬 **Built-in Chat**: Conversational interface with context-aware responses
- 📚 **PDF Support**: Direct import and indexing of PDF documents
- 🚀 **Fast Retrieval**: Sub-second search across massive datasets
- 💾 **Efficient Storage**: 10x compression compared to traditional databases
- 🔌 **Pluggable LLMs**: Works with OpenAI, Anthropic, Google, OpenRouter, Groq, or local models
- 🌐 **Offline-First**: No internet required after video generation
- 🔧 **Simple API**: Get started with just 3 lines of code

## 🎯 Use Cases

- **📖 Digital Libraries**: Index thousands of books in a single video file
- **🎓 Educational Content**: Create searchable video memories of course materials
- **📰 News Archives**: Compress years of articles into manageable video databases
- **💼 Corporate Knowledge**: Build company-wide searchable knowledge bases
- **🔬 Research Papers**: Quick semantic search across scientific literature
- **📝 Personal Notes**: Transform your notes into a searchable AI assistant

## 🚀 Why Memvid?

### Game-Changing Innovation
- **Video as Database**: Store millions of text chunks in a single MP4 file
- **Instant Retrieval**: Sub-second semantic search across massive datasets
- **10x Storage Efficiency**: Video compression reduces memory footprint dramatically
- **Zero Infrastructure**: No database servers, just files you can copy anywhere
- **Offline-First**: Works completely offline once videos are generated

### Lightweight Architecture
- **Minimal Dependencies**: Core functionality in ~1000 lines of Python
- **CPU-Friendly**: Runs efficiently without GPU requirements
- **Portable**: Single video file contains your entire knowledge base
- **Streamable**: Videos can be streamed from cloud storage

## 📦 Installation

### Quick Install
```bash
pip install memvid
```

### For PDF Support
```bash
pip install memvid PyPDF2
```

### For Enhanced PDF & OCR Support
```bash
# Standard PDF processing
pip install memvid PyPDF2 pymupdf

# OCR for scanned/handwritten PDFs  
pip install memvid PyPDF2 pymupdf pytesseract easyocr

# All PDF features
pip install "memvid[pdf]"
```

### Recommended Setup (Virtual Environment)
```bash
# Create a new project directory
mkdir my-memvid-project
cd my-memvid-project

# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install memvid
pip install memvid

# For PDF support:
pip install PyPDF2
```

## 🎯 Quick Start

### Basic Usage
```python
from memvid import MemvidEncoder, MemvidChat

# Create video memory from text chunks
chunks = ["Important fact 1", "Important fact 2", "Historical event details"]
encoder = MemvidEncoder()
encoder.add_chunks(chunks)
encoder.build_video("memory.mp4", "memory_index.json")

# Chat with your memory
chat = MemvidChat("memory.mp4", "memory_index.json")
chat.start_session()
response = chat.chat("What do you know about historical events?")
print(response)
```

### Building Memory from Documents
```python
from memvid import MemvidEncoder
import os

# Load documents
encoder = MemvidEncoder(chunk_size=512, overlap=50)

# Add text files
for file in os.listdir("documents"):
    with open(f"documents/{file}", "r") as f:
        encoder.add_text(f.read(), metadata={"source": file})

# Build optimized video
encoder.build_video(
    "knowledge_base.mp4",
    "knowledge_index.json",
    fps=30,  # Higher FPS = more chunks per second
    frame_size=512  # Larger frames = more data per frame
)
```

### Advanced Search & Retrieval
```python
from memvid import MemvidRetriever

# Initialize retriever
retriever = MemvidRetriever("knowledge_base.mp4", "knowledge_index.json")

# Semantic search
results = retriever.search("machine learning algorithms", top_k=5)
for chunk, score in results:
    print(f"Score: {score:.3f} | {chunk[:100]}...")

# Get context window
context = retriever.get_context("explain neural networks", max_tokens=2000)
print(context)
```

### Interactive Chat Interface
```python
from memvid import MemvidInteractive

# Launch interactive chat UI
interactive = MemvidInteractive("knowledge_base.mp4", "knowledge_index.json")
interactive.run()  # Opens web interface at http://localhost:7860
```

### Testing with file_chat.py
The `examples/file_chat.py` script provides a comprehensive way to test Memvid with your own documents:

```bash
# Process a directory of documents
python examples/file_chat.py --input-dir /path/to/documents --provider google

# Process specific files
python examples/file_chat.py --files doc1.txt doc2.pdf --provider openai

# Use H.265 compression (requires Docker)
python examples/file_chat.py --input-dir docs/ --codec h265 --provider google

# Custom chunking for large documents
python examples/file_chat.py --files large.pdf --chunk-size 2048 --overlap 32 --provider google

# Load existing memory
python examples/file_chat.py --load-existing output/my_memory --provider google
```

### Complete Example: Chat with a PDF Book
```bash
# 1. Create a new directory and set up environment
mkdir book-chat-demo
cd book-chat-demo
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install memvid PyPDF2

# 3. Create book_chat.py
cat > book_chat.py << 'EOF'
from memvid import MemvidEncoder, chat_with_memory
import os

# Your PDF file
book_pdf = "book.pdf"  # Replace with your PDF path

# Build video memory
encoder = MemvidEncoder()
encoder.add_pdf(book_pdf)
encoder.build_video("book_memory.mp4", "book_index.json")

# Chat with the book
api_key = os.getenv("OPENAI_API_KEY")  # Optional: for AI responses
chat_with_memory("book_memory.mp4", "book_index.json", api_key=api_key)
EOF

# 4. Run it
export OPENAI_API_KEY="your-api-key"  # Optional
python book_chat.py
```

## 📄 Advanced PDF Processing

Memvid supports multiple PDF processing methods to handle different types of PDF documents:

### Supported PDF Processors

| Processor | Best For | Performance | Dependencies |
|-----------|----------|-------------|--------------|
| **pypdf2** | Digital PDFs with selectable text | Fast | `PyPDF2` |
| **pymupdf** | Better text extraction from digital PDFs | Fast | `pymupdf` |
| **ocr_tesseract** | Scanned PDFs, images with text | Medium | `pymupdf`, `pytesseract`, `Pillow` |
| **ocr_easyocr** | Handwritten text, multilingual content | Slow | `pymupdf`, `easyocr`, `numpy`, `Pillow` |
| **ocr_handwritten** | Specialized for prescriptions, forms | Slowest | All OCR dependencies + `opencv-python` |

### Usage Examples

```python
from memvid import MemvidEncoder

encoder = MemvidEncoder()

# Standard digital PDF (default)
encoder.add_pdf("document.pdf", pdf_processor="pypdf2")

# Better extraction for digital PDFs
encoder.add_pdf("document.pdf", pdf_processor="pymupdf")

# OCR for scanned documents
encoder.add_pdf("scanned.pdf", pdf_processor="ocr_tesseract")

# OCR optimized for handwritten text
encoder.add_pdf("handwritten.pdf", pdf_processor="ocr_easyocr")

# Specialized processing for prescriptions and complex handwriting
encoder.add_pdf("prescription.pdf", pdf_processor="ocr_handwritten")
```

### Command Line Usage

```bash
# Use enhanced PDF extraction
python examples/file_chat.py --files document.pdf --pdf-processor pymupdf

# Process scanned PDFs with OCR
python examples/file_chat.py --files scanned.pdf --pdf-processor ocr_tesseract

# Handle handwritten documents
python examples/file_chat.py --files notes.pdf --pdf-processor ocr_easyocr

# Specialized processing for prescriptions and complex handwriting
python examples/file_chat.py --files prescription.pdf --pdf-processor ocr_handwritten

# Compare different processors
python examples/test_pdf_processors.py your_document.pdf
```

### Installation for PDF Features

```bash
# Install all PDF processing capabilities
pip install "memvid[pdf]"

# Or install specific components
pip install pymupdf                    # Enhanced PDF extraction
pip install pytesseract Pillow        # Tesseract OCR
pip install easyocr numpy             # EasyOCR for handwritten text
```

### Performance Comparison

When choosing a PDF processor, consider:

- **Digital PDFs**: Use `pymupdf` for best results (better than `pypdf2`)
- **Scanned documents**: Use `ocr_tesseract` for good accuracy and speed
- **Handwritten text**: Use `ocr_easyocr` for better handwriting recognition
- **Complex handwriting/prescriptions**: Use `ocr_handwritten` for maximum accuracy
- **Multilingual content**: Use `ocr_easyocr` with language support

### Testing PDF Processors

Use the comparison tool to find the best processor for your documents:

```bash
# Test all available processors
python examples/test_pdf_processors.py sample.pdf

# Test specific processors
python examples/test_pdf_processors.py scanned.pdf --processors ocr_tesseract ocr_easyocr
```

The tool will show you:
- Processing speed comparison
- Text extraction quality
- Recommendations for your document type

## 🛠️ Advanced Configuration

### Custom Embeddings
```python
from sentence_transformers import SentenceTransformer

# Use custom embedding model
custom_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
encoder = MemvidEncoder(embedding_model=custom_model)
```

### Video Optimization
```python
# For maximum compression
encoder.build_video(
    "compressed.mp4",
    "index.json",
    fps=60,  # More frames per second
    frame_size=256,  # Smaller frames
    video_codec='h265',  # Better compression
    crf=28  # Compression quality (lower = better quality)
)
```

### Distributed Processing
```python
# Process large datasets in parallel
encoder = MemvidEncoder(n_workers=8)
encoder.add_chunks_parallel(massive_chunk_list)
```

## 🤖 LLM Provider Support

Memvid supports multiple LLM providers for AI-powered conversations with your memory:

### Supported Providers

| Provider | Models | API Key Required | Notes |
|----------|--------|------------------|-------|
| **Google** | gemini-2.0-flash-exp, gemini-1.5-pro | `GOOGLE_API_KEY` | Default provider, fast and reliable |
| **OpenAI** | gpt-4o, gpt-4, gpt-3.5-turbo | `OPENAI_API_KEY` | Industry standard, excellent quality |
| **Anthropic** | claude-3.5-sonnet, claude-3-haiku | `ANTHROPIC_API_KEY` | Advanced reasoning capabilities |
| **OpenRouter** | 100+ models via unified API | `OPENROUTER_API_KEY` | Access to multiple providers through one API (default: free model) |
| **Groq** | llama3-70b-8192, mixtral-8x7b | `GROQ_API_KEY` | Ultra-fast inference speeds |

### Usage Examples

```python
from memvid import MemvidChat, LLMClient

# Using Google (default)
chat = MemvidChat("memory.mp4", "index.json", llm_provider="google")

# Using OpenAI
chat = MemvidChat("memory.mp4", "index.json", llm_provider="openai", llm_model="gpt-4o")

# Using Anthropic Claude
chat = MemvidChat("memory.mp4", "index.json", llm_provider="anthropic", llm_model="claude-3.5-sonnet")

# Using OpenRouter (access to many models)
chat = MemvidChat("memory.mp4", "index.json", llm_provider="openrouter", llm_model="google/gemini-2.0-flash-exp:free")

# Using Groq (ultra-fast)
chat = MemvidChat("memory.mp4", "index.json", llm_provider="groq", llm_model="llama3-70b-8192")

# Test standalone LLM clients
client = LLMClient(provider="openrouter", model="meta-llama/llama-3.1-70b-instruct")
response = client.chat([{"role": "user", "content": "Hello!"}])
```

### Environment Variables

Set your API keys as environment variables:

```bash
# Google AI
export GOOGLE_API_KEY="AIzaSyB..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenRouter (access to 100+ models)
export OPENROUTER_API_KEY="sk-or-..."

# Groq (ultra-fast inference)
export GROQ_API_KEY="gsk_..."
```

### Provider-Specific Features

**OpenRouter Benefits:**
- Access to 100+ models from different providers
- Unified API for switching between models
- Cost-effective pricing
- Real-time model availability

**Groq Benefits:**
- Lightning-fast inference (up to 500 tokens/second)
- Optimized for Llama and Mixtral models
- Low latency for real-time applications

### Command Line Usage

Use providers with the file_chat example:

```bash
# Default Google
python examples/file_chat.py --input-dir docs/ --provider google

# OpenAI GPT-4
python examples/file_chat.py --input-dir docs/ --provider openai --model gpt-4o

# Anthropic Claude
python examples/file_chat.py --input-dir docs/ --provider anthropic --model claude-3.5-sonnet

# OpenRouter (access many models)
python examples/file_chat.py --input-dir docs/ --provider openrouter --model "google/gemini-2.0-flash-exp:free"

# Groq (ultra-fast)
python examples/file_chat.py --input-dir docs/ --provider groq --model llama3-70b-8192
```

### Testing New Providers

Use the test script to verify your setup:

```bash
# Test all available providers
python examples/test_new_providers.py

# Set API keys first
export OPENROUTER_API_KEY="your-key"
export GROQ_API_KEY="your-key"
python examples/test_new_providers.py
```

## 🐛 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'memvid'**
```bash
# Make sure you're using the right Python
which python  # Should show your virtual environment path
# If not, activate your virtual environment:
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**ImportError: PyPDF2 is required for PDF support**
```bash
pip install PyPDF2
```

**LLM API Key Issues**
```bash
# Set your API key (get one at https://platform.openai.com)
export GOOGLE_API_KEY="AIzaSyB1-..."  # macOS/Linux
# Or on Windows:
set GOOGLE_API_KEY=AIzaSyB1-...
```

**Large PDF Processing**
```python
# For very large PDFs, use smaller chunk sizes
encoder = MemvidEncoder()
encoder.add_pdf("large_book.pdf", chunk_size=400, overlap=50)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=memvid tests/

# Format code
black memvid/
```

## 🆚 Comparison with Traditional Solutions

| Feature | Memvid | Vector DBs | Traditional DBs |
|---------|--------|------------|-----------------|
| Storage Efficiency | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Setup Complexity | Simple | Complex | Complex |
| Semantic Search | ✅ | ✅ | ❌ |
| Offline Usage | ✅ | ❌ | ✅ |
| Portability | File-based | Server-based | Server-based |
| Scalability | Millions | Millions | Billions |
| Cost | Free | $$$$ | $$$ |


## 📚 Examples

Check out the [examples/](examples/) directory for:
- Building memory from Wikipedia dumps
- Creating a personal knowledge base
- Multi-language support
- Real-time memory updates
- Integration with popular LLMs

## 🆘 Getting Help

- 📖 [Documentation](https://github.com/olow304/memvid/wiki) - Comprehensive guides
- 💬 [Discussions](https://github.com/olow304/memvid/discussions) - Ask questions
- 🐛 [Issue Tracker](https://github.com/olow304/memvid/issues) - Report bugs
- 🌟 [Show & Tell](https://github.com/olow304/memvid/discussions/categories/show-and-tell) - Share your projects

## 🔗 Links

- [GitHub Repository](https://github.com/olow304/memvid)
- [PyPI Package](https://pypi.org/project/memvid)
- [Changelog](https://github.com/olow304/memvid/releases)


## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Created by [Olow304](https://github.com/olow304) and the Memvid community.

Built with ❤️ using:
- [sentence-transformers](https://www.sbert.net/) - State-of-the-art embeddings for semantic search
- [OpenCV](https://opencv.org/) - Computer vision and video processing
- [qrcode](https://github.com/lincolnloop/python-qrcode) - QR code generation
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [PyPDF2](https://github.com/py-pdf/pypdf) - PDF text extraction

Special thanks to all contributors who help make Memvid better!

---

**Ready to revolutionize your AI memory management? Install Memvid and start building!** 🚀
