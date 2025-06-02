# Memvid - Efficient AI Memory with SQLite & FAISS 🧠💾

**A lightweight Python library for creating fast, local, and searchable AI memories using SQLite and FAISS.**

[![PyPI version](https://badge.fury.io/py/memvid.svg)](https://pypi.org/project/memvid/)
[![Downloads](https://pepy.tech/badge/memvid)](https://pepy.tech/project/memvid)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Memvid helps you build searchable AI memories from your text data. It stores text chunks efficiently in an SQLite database (with zlib compression) and uses FAISS for lightning-fast semantic search. This approach provides a portable, serverless, and offline-first knowledge base solution.

## ✨ Key Features

- 💾 **SQLite Backend**: Stores text chunks in a portable SQLite database.
- 🗜️ **Text Compression**: Chunks are compressed (zlib) in the database.
- 🔍 **Semantic Search**: Find relevant content using natural language queries via FAISS.
- 💬 **Built-in Chat**: Interactive chat interface (`chat_with_memory`) to converse with your data.
- 📚 **PDF Support**: Direct import and indexing of PDF documents.
- 🚀 **Fast Retrieval**: Optimized for quick access to relevant information.
- 🔌 **Pluggable LLMs**: Works with OpenAI (or can run context-only) for chat responses.
- 🌐 **Offline-First**: Memory (DB + Index files) can be used entirely offline after creation.
- 🔧 **Simple API**: Easy to integrate into your Python projects.
- 📦 **Portable**: Memory consists of a database file and a few index files.

## 🎯 Use Cases

- **📖 Digital Libraries**: Index thousands of books in a single portable database.
- **🎓 Educational Content**: Create searchable knowledge bases from course materials.
- **📰 News Archives**: Compress years of articles into manageable databases.
- **💼 Corporate Knowledge**: Build company-wide searchable knowledge bases.
- **🔬 Research Papers**: Quick semantic search across scientific literature.
- **📝 Personal Notes**: Transform your notes into a searchable AI assistant.

## 🚀 Why Memvid?

### Core Advantages
- **Efficient Text Storage**: Combines SQLite with zlib compression for text chunks.
- **Fast Semantic Search**: Leverages FAISS for quick and relevant information retrieval.
- **Zero Infrastructure**: No database servers needed; just local files.
- **Portable Knowledge**: Your entire memory (database + index) is easily transferable.
- **Offline Capable**: Access your created memories without an internet connection.

### Lightweight Architecture
- **Minimal Dependencies**: Core functionality in Python. SQLite and zlib are standard. Key dependencies are `sentence-transformers`, `faiss-cpu`. `PyPDF2` is optional for PDF support. `python-dotenv` is used in examples.
- **CPU-Friendly**: Runs efficiently without GPU requirements for CPU version of FAISS.

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

# Install memvid from PyPI (for regular users)
pip install memvid

# For PDF support (optional, if using memvid as a library and need PDF functionality):
pip install PyPDF2 # Or pip install memvid[pdf] if setup.py extras_require is used by pip install memvid

### Development Setup (for contributors or running examples from source)
If you have cloned the repository and want to develop `memvid` or run examples from the source code:

```bash
# (Ensure you are in the root of the cloned memvid project)
# (And your virtual environment is activated, as shown above)

# Install in development mode (editable install)
# This installs core dependencies from setup.py's install_requires
pip install -e .

# To include development tools (like pytest, black):
pip install -e ".[dev]"

# To include optional PDF support:
pip install -e ".[pdf]"
# (This assumes setup.py has extras_require for "pdf" like: extras_require={"pdf": ["PyPDF2"]})
# Alternatively, install PyPDF2 directly: pip install PyPDF2

# Or, to install all dependencies listed in requirements.txt (for a consistent environment):
# pip install -r requirements.txt
```
*(Note: `extras_require` for "pdf" is already defined in `setup.py`)*

### Running Examples Locally
The scripts in the `examples/` directory use `sys.path.insert` at the top. This technique allows the Python interpreter to find the local `memvid` module code within the repository *without* the `memvid` library itself being formally installed into the site-packages of the virtual environment in the traditional sense (if you haven't run `pip install .` yet).

However, `sys.path.insert` does **not** install the library's dependencies (like `tqdm`, `sentence-transformers`, `faiss-cpu`, etc.) into your Python environment.

To run the examples successfully, ensure you have first installed `memvid` and its dependencies from the project's root directory within your activated virtual environment:
```bash
# From the root of the memvid project, after activating your virtual environment:
pip install -e .
# This installs core dependencies. For optional ones like PDF, add them as needed:
# pip install -e ".[pdf]"
# pip install python-dotenv # (dotenv is used by some examples for API keys)
```
Alternatively, `pip install -r requirements.txt` can be used if it's kept up-to-date with all necessary packages for examples.

After installation, you can then run example scripts like:
```bash
python examples/build_memory.py
python examples/simple_chat.py
```

## 🎯 Quick Start

### Basic Usage
```python
# Ensure you have installed memvid and its dependencies first
# (e.g., by running `pip install -e .` from the project root in your venv)
from memvid import MemvidEncoder, MemvidChat
from memvid.config import get_default_config # For config if customizing DB path
import os

# --- 1. Build Memory ---
# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Define paths for database and index files
db_file = os.path.join(output_dir, "my_memory.db")
index_prefix = os.path.join(output_dir, "my_memory_index") # For my_memory_index.faiss, etc.

# Configure encoder to use the specified database path
config = get_default_config()
config["database"]["path"] = db_file

encoder = MemvidEncoder(config=config)

chunks = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is rapidly evolving.",
    "SQLite is a lightweight, file-based database."
]
encoder.add_chunks(chunks) # Can also use add_text or add_pdf

print(f"Building memory with DB: {db_file}, Index prefix: {index_prefix}")
encoder.build_memory(index_file_path_prefix=index_prefix)
print("Memory built.")

# --- 2. Chat with Memory ---
# Ensure config for chat points to the same DB.
# MemvidChat passes its config to MemvidRetriever, which gets database.path.
chat = MemvidChat(index_file_path_prefix=index_prefix, config=config) # Pass the same config
chat.start_session()
response = chat.chat("What do you know about databases?")
print(f"Response: {response}")
```

### Building Memory from Documents
```python
from memvid import MemvidEncoder
from memvid.config import get_default_config
import os

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
db_file = os.path.join(output_dir, "docs_memory.db")
index_prefix = os.path.join(output_dir, "docs_memory_index")

config = get_default_config()
config["database"]["path"] = db_file

encoder = MemvidEncoder(config=config)

# Example: Add text files (replace with your actual document loading)
# for file_name in os.listdir("path_to_your_documents"):
#     if file_name.endswith(".txt"): # Example filter
#         with open(os.path.join("path_to_your_documents", file_name), "r", encoding='utf-8') as f:
#             encoder.add_text(f.read(), metadata={"source": file_name})

encoder.add_text("Sample document text about technology.", metadata={"source": "doc1.txt"})
encoder.add_text("Another document discussing AI advancements.", metadata={"source": "doc2.txt"})

encoder.build_memory(index_file_path_prefix=index_prefix)
print(f"Memory built from documents. DB: {db_file}, Index files prefix: {index_prefix}")
```

### Advanced Search & Retrieval
```python
from memvid import MemvidRetriever
from memvid.config import get_default_config
import os

output_dir = "output"
# Ensure these paths match how your memory was built
db_file = os.path.join(output_dir, "my_memory.db") # Example DB name from Quick Start
index_prefix = os.path.join(output_dir, "my_memory_index")

config = get_default_config()
config["database"]["path"] = db_file # Crucial: ensure retriever uses the correct DB

retriever = MemvidRetriever(index_file_path_prefix=index_prefix, config=config)

results_text = retriever.search("artificial intelligence", top_k=2)
for text_chunk in results_text:
    print(f"Found: {text_chunk[:100]}...")

# For results with metadata
results_meta = retriever.search_with_metadata("SQLite database", top_k=1)
for res_dict in results_meta:
    print(f"Text: {res_dict['text'][:50]}..., Score: {res_dict['score']:.3f}, ChunkID: {res_dict['chunk_id']}")

# get_context_window example
if results_meta:
    first_chunk_id_from_search = results_meta[0]['chunk_id']
    context = retriever.get_context_window(first_chunk_id_from_search, window_size=1)
    print(f"\nContext window for chunk {first_chunk_id_from_search}: {context}")
else:
    print("\nRun search for 'SQLite database' to get a chunk ID for context window example.")
```

### Interactive Chat Interface
```python
from memvid import chat_with_memory # Helper function from memvid.interactive
from memvid.config import get_default_config
import os

output_dir = "output"
# This should match the prefix used when building the memory
index_prefix = os.path.join(output_dir, "my_memory_index")
# This should be the database file associated with the index_prefix
db_file_for_chat = os.path.join(output_dir, "my_memory.db")

# Prepare config for chat_with_memory to point to the correct DB
chat_config = get_default_config()
chat_config["database"]["path"] = db_file_for_chat

print(f"Ensure memory files exist with prefix: {index_prefix}")
print(f"And associated database '{db_file_for_chat}' is present.")

# Call with the index prefix and the config specifying the DB path
chat_with_memory(index_file_path_prefix=index_prefix, config=chat_config)
```

### Complete Example: Chat with a PDF Book
```bash
# 1. Create a new directory and set up environment
mkdir book-chat-demo
cd book-chat-demo
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install memvid PyPDF2 python-dotenv # Add python-dotenv for API key management

# 3. Create book_chat.py (see examples/book_chat.py for the full updated script)
#    It will define a DB path and index_prefix, e.g., output/book_memory.db and output/book_memory_index
#    Then it calls:
#    encoder.add_pdf(book_pdf_path)
#    encoder.build_memory(index_file_path_prefix)
#    chat_with_memory(index_file_path_prefix, config=...) # Config points to the DB

# 4. (Optional) Create a .env file for your OpenAI API key
# echo 'OPENAI_API_KEY="your-sk-key-here"' > .env

# 5. Run it (ensure you have a PDF, e.g. data/bitcoin.pdf from the original repo examples)
# mkdir data
# # (put bitcoin.pdf or your_book.pdf into data directory)
# python examples/book_chat.py # (Or your script name)
```
*(For the full `book_chat.py` script, please refer to the `examples` directory in the repository, as it's more detailed than what can be concisely shown here.)*

## 🔧 API Reference

### `MemvidEncoder`
Handles chunking text, adding to the database, and building the FAISS index.
- `__init__(self, config: Optional[Dict[str, Any]] = None)`: Initializes with an optional configuration. Config can specify `database.path`, `embedding.model`, etc.
- `add_chunks(self, chunks: List[str], metadata_list: Optional[List[Dict[str, Any]]]=None)`: Adds pre-chunked text with optional metadata per chunk.
- `add_text(self, text: str, chunk_size: int = 500, overlap: int = 50, metadata: Optional[Dict[str, Any]] = None)`: Chunks the given text and adds it with common metadata.
- `add_pdf(self, pdf_path: str, chunk_size: int = 800, overlap: int = 100)`: Extracts text from a PDF, chunks it, and adds it. Metadata includes PDF source.
- `build_memory(self, index_file_path_prefix: str, show_progress: bool = True)`: Builds the FAISS index from chunks in the database and saves it to files starting with `index_file_path_prefix`.
- `get_stats(self) -> Dict[str, Any]`: Returns statistics about the encoder and database.
- `clear(self)`: Clears all chunks from the database and resets the index manager.

### `MemvidRetriever`
Retrieves information from the memory (database + FAISS index).
- `__init__(self, index_file_path_prefix: str, config: Optional[Dict[str, Any]] = None)`: Initializes with the path prefix for index files and an optional configuration (which should specify `database.path`).
- `search(self, query: str, top_k: int = 5) -> List[str]`: Performs semantic search and returns a list of relevant text chunks.
- `search_with_metadata(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]`: Returns search results with text, score, chunk ID, and metadata.
- `get_chunk_by_id(self, chunk_id: int) -> Optional[str]`: Retrieves the text content of a specific chunk by its ID.
- `get_context_window(self, chunk_id: int, window_size: int = 2) -> List[str]`: Retrieves a chunk and its surrounding chunks by ID.
- `get_stats(self) -> Dict[str, Any]`: Returns statistics about the retriever, cache, and index.
- `clear_cache(self)`: Clears the retriever's internal chunk cache.

### `MemvidChat`
Manages conversational interactions using the memory.
- `__init__(self, index_file_path_prefix: str, llm_api_key: Optional[str] = None, llm_model: Optional[str] = None, config: Optional[Dict[str, Any]] = None)`: Initializes with index prefix, optional LLM details, and configuration.
- `chat(self, user_input: str) -> str`: Processes user input, retrieves context, (optionally) calls an LLM, and returns a response.
- `search_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]`: Performs a search and returns results with metadata, without LLM interaction.
- (Other methods: `start_session`, `get_history`, `export_session`, `load_session`, `reset_session`, `get_stats`)

### `memvid.interactive.chat_with_memory`
A helper function to quickly start an interactive command-line chat session.
- `chat_with_memory(index_file_path_prefix: str, api_key: Optional[str] = None, llm_model: Optional[str] = None, config: Optional[Dict[str, Any]] = None, ...)`

## 🛠️ Advanced Configuration

Memvid's behavior can be customized via a configuration dictionary passed to its main classes (`MemvidEncoder`, `MemvidRetriever`, `MemvidChat`). See `memvid/config.py` for default values.

Key configuration sections:
- `database`:
    - `path`: Path to the SQLite database file (e.g., `"memvid_data/main_memory.db"`).
- `embedding`:
    - `model`: Name of the SentenceTransformer model (e.g., `"all-MiniLM-L6-v2"`).
    - `dimension`: Output dimension of the embedding model.
- `index`:
    - `type`: FAISS index type (e.g., `"Flat"`, `"IVF"`).
    - `nlist`: Number of clusters for IVF index type.
- `retrieval`:
    - `batch_size`: Batch size for encoding embeddings during index building.
    - `cache_size`: (Note: `MemvidRetriever` currently uses a hardcoded `lru_cache(maxsize=1000)`. This config key is for future use if cache becomes more configurable).
- `llm`: (For `MemvidChat`)
    - `model`: Default LLM model name (e.g., `"gpt-3.5-turbo"`).
    - `max_tokens`, `temperature`, `context_window`.
- `chat`: (For `MemvidChat`)
    - `context_chunks`: Number of context chunks to retrieve per query.
    - `max_history`: Number of past messages to include in LLM prompt.

Example:
```python
from memvid import MemvidEncoder, get_default_config

custom_config = get_default_config()
custom_config["database"]["path"] = "my_data/project_alpha.db"
custom_config["embedding"]["model"] = "sentence-transformers/all-mpnet-base-v2"
custom_config["embedding"]["dimension"] = 768 # Must match the chosen model
custom_config["index"]["type"] = "IVF"
custom_config["index"]["nlist"] = 256

encoder = MemvidEncoder(config=custom_config)
# ... then use encoder, retriever, chat with this config.
```

## 🐛 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'memvid'**
Ensure your virtual environment is activated and `memvid` is installed (`pip show memvid`). If running from cloned repo, ensure the project root is in `PYTHONPATH`.

**ImportError: PyPDF2 is required for PDF support**
Install it: `pip install PyPDF2`

**FAISS Index File Not Found (e.g., `my_index.faiss`)**
Ensure you have run the `encoder.build_memory("my_index")` step. This creates the `.faiss` file and an associated `.indexinfo.json` file. Both are needed by `MemvidRetriever` and `MemvidChat`.

**Database File Not Found**
The database file (e.g., `my_memory.db`) is created by `MemvidEncoder` when data is first added. Ensure the path specified in the `config["database"]["path"]` is correct and accessible for both encoding and retrieval/chat.

**OpenAI API Key Issues**
Set the `OPENAI_API_KEY` environment variable or pass the `llm_api_key` argument to `MemvidChat`.

**Large PDF Processing**
For very large PDFs, ensure sufficient memory. `add_pdf` processes text page by page. Chunking parameters in `add_pdf` or `add_text` can be adjusted.

## 🤝 Contributing

We welcome contributions! Please see our (future) `CONTRIBUTING.md` for details. For now, feel free to open issues or suggest features.

```bash
# Run tests (ensure pytest and necessary mocks are installed)
pytest tests/

# Run with coverage (ensure pytest-cov is installed)
pytest --cov=memvid tests/

# Format code (ensure black is installed)
black memvid/ tests/ examples/
```

## 🆚 Comparison with Traditional Solutions

| Feature | Memvid (SQLite+FAISS) | Vector DBs (Server) | Traditional DBs (SQL) |
|---------|-----------------------|---------------------|-----------------------|
| Storage Efficiency | ⭐⭐⭐⭐ (Text + Compressed Chunks + FAISS index) | ⭐⭐ (Embeddings are large) | ⭐⭐⭐ (Text only) |
| Setup Complexity | Simple (pip install) | Complex (Server setup) | Moderate to Complex |
| Semantic Search | ✅ (FAISS) | ✅ (Core feature) | ❌ (Requires extensions) |
| Offline Usage | ✅ (Local files) | Often ❌ (Server needed) | ✅ (Local instances) / ❌ (Cloud) |
| Portability | ✅ (DB file + Index files) | Harder (DB dumps, server migration) | Harder |
| Scalability (Data Vol.)| Millions of chunks (practical limits depend on disk/RAM for FAISS) | Billions+ | Billions+ |
| Cost (Infra) | Free (local files) | $$$ (Server hosting) | $$-$$$ |
| Dependencies | Minimal Python libs | External server process | External server process (often) |


## 🗺️ Roadmap

- [ ] **v0.2.0** - Multi-language embedding support evaluation.
- [ ] **v0.3.0** - More robust metadata filtering in search.
- [ ] **v0.4.0** - Explore options for dynamic index updates (without full rebuild).
- [ ] **v0.5.0** - Potential for audio/image transcript/description indexing.
- [ ] **v1.0.0** - Production-ready with more comprehensive documentation and examples.

## 📚 Examples

Check out the [examples/](examples/) directory for:
- `build_memory.py`: Basic script to create a memory from text chunks.
- `simple_chat.py`: Demonstrates `quick_chat` and `chat_with_memory`.
- `chat_memory.py` / `chat_memory_fixed.py`: Full interactive chat example.
- `book_chat.py`: Example of building memory from a PDF and chatting with it.
- `soccer_chat.py`: Example with a specific small dataset.

## 🆘 Getting Help

- 📖 [Documentation](README.md) - This file!
- 🐛 [Issue Tracker](https://github.com/olow304/memvid/issues) - Report bugs or ask questions. *(Assuming this is the correct repo URL based on original README)*

## 🔗 Links

- [GitHub Repository](https://github.com/olow304/memvid) *(Assuming)*
- [PyPI Package](https://pypi.org/project/memvid)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Original concept inspired by the need for simple, local AI memory.

Built with ❤️ using:
- [sentence-transformers](https://www.sbert.net/) - State-of-the-art embeddings.
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search.
- [SQLite](https://www.sqlite.org/) (via Python's `sqlite3`) - Database backend.
- [zlib](https://www.zlib.net/) (via Python's `zlib`) - Text compression.
- [PyPDF2](https://github.com/py-pdf/pypdf) - PDF text extraction (optional).
- Other great Python libraries!

Special thanks to all contributors who help make Memvid better!

---

**Ready to build your own local, searchable AI memories? Install Memvid and start experimenting!** 🚀
