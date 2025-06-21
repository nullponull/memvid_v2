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
- 🔌 **Pluggable LLMs**: Works with OpenAI, Anthropic, or local models
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

---

# BitMatrix⇔JSON変換機能 (日本語)

## 概要
MemVidにBitMatrix⇔JSON変換機能が追加されました。この機能により、QRコードの画像ファイル（PNG）を読み込む代わりに、軽量なJSONファイルから直接データを取得できるようになり、大幅なパフォーマンス向上を実現します。

## 主な改善点
- **メモリ使用量の削減**: 8GB超から約200MBに激減
- **検索遅延の最小化**: 従来比約10%増（900ms vs 820ms）のみ
- **I/O効率の向上**: 画像読み込み・デコード処理を省略

## 使用方法

### 基本的な使い方
```python
from memvid import MemvidEncoder, MemvidRetriever

# エンコード（自動的にPNGとJSONの両方が生成されます）
encoder = MemvidEncoder()
encoder.add_chunks(["テキストチャンク1", "テキストチャンク2"])
encoder.build_video("memory.mp4", "memory_index.json")

# 検索（JSONファイルが利用可能な場合は自動的に使用されます）
retriever = MemvidRetriever("memory.mp4", "memory_index.json")
results = retriever.search("検索クエリ", top_k=5)
```

### パフォーマンス最適化
システムは以下の順序でデータを取得します：
1. **キャッシュ**: メモリ内キャッシュから取得
2. **JSON**: 軽量なJSONファイルから直接読み込み
3. **画像**: フォールバックとして動画からQRコードを抽出・デコード

### 後方互換性
- 既存の動画ファイルは引き続き動作します
- JSONファイルが存在しない場合は自動的に従来の画像デコードにフォールバックします
- 新しく生成される動画では、PNGとJSONの両方のファイルが作成されます

## 技術詳細

### データフロー
1. **エンコード時**: テキストチャンク → JSON文字列 → QRコード → PNG画像 + BitMatrix JSON
2. **検索時**: クエリ → セマンティック検索 → フレーム番号 → JSON読み込み → テキスト取得

### ファイル構造
```
frames/
├── frame_000000.png  # 従来の画像ファイル（後方互換性用）
├── frame_000000.json # 新しいBitMatrixデータ
├── frame_000001.png
├── frame_000001.json
└── ...
```

### 圧縮処理
- 100文字を超える長いテキストは自動的にgzip圧縮 + Base64エンコードされます
- 圧縮されたデータには"GZ:"プレフィックスが付与されます
- JSONファイルには元のデータと圧縮データの両方が保存されます

### 使用例：大量文書の処理
```python
from memvid import MemvidEncoder, MemvidRetriever
import os

# 大量の文書を処理
encoder = MemvidEncoder(chunk_size=512, overlap=50)

# PDFファイルを追加
for pdf_file in os.listdir("documents"):
    if pdf_file.endswith(".pdf"):
        encoder.add_pdf(f"documents/{pdf_file}")

# 最適化された動画を構築
encoder.build_video(
    "knowledge_base.mp4",
    "knowledge_index.json",
    codec="h265"  # 高圧縮率
)

# 高速検索
retriever = MemvidRetriever("knowledge_base.mp4", "knowledge_index.json")
results = retriever.search("機械学習アルゴリズム", top_k=10)

# 結果の表示
for i, result in enumerate(results):
    print(f"{i+1}. {result[:100]}...")
```</str>
