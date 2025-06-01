import sqlite3
import json
import zlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Default database file name, can be overridden in config
DEFAULT_DB_FILENAME = "memvid_memory.db"

class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or DEFAULT_DB_FILENAME)
        self._ensure_db_directory()
        self._init_db()

    def _ensure_db_directory(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10) # Increased timeout
        conn.row_factory = sqlite3.Row # Access columns by name
        return conn

    def _init_db(self):
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_content TEXT, -- Store original for debugging/future, or make optional
                    compressed_content BLOB NOT NULL,
                    original_length INTEGER NOT NULL, -- Length of uncompressed text_content
                    metadata TEXT -- JSON encoded metadata
                )
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS info (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """)
                # Potentially store schema version or other global info
                cursor.execute("INSERT OR IGNORE INTO info (key, value) VALUES (?, ?)",
                               ("schema_version", "1.0"))
                conn.commit()
            logger.info(f"Database initialized/verified at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def add_chunk(self, text_content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        if not text_content:
            raise ValueError("Text content cannot be empty.")

        compressed_content = zlib.compress(text_content.encode('utf-8'))
        original_length = len(text_content)
        metadata_json = json.dumps(metadata or {})

        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO chunks (text_content, compressed_content, original_length, metadata)
                VALUES (?, ?, ?, ?)
                """, (text_content, compressed_content, original_length, metadata_json))
                conn.commit()
                chunk_id = cursor.lastrowid
                logger.debug(f"Added chunk {chunk_id} (original length: {original_length}, compressed: {len(compressed_content)})")
                return chunk_id
        except sqlite3.Error as e:
            logger.error(f"Failed to add chunk: {e}")
            raise

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id, text_content, compressed_content, original_length, metadata FROM chunks WHERE chunk_id = ?", (chunk_id,))
                row = cursor.fetchone()

                if row:
                    text_content_decompressed = zlib.decompress(row["compressed_content"]).decode('utf-8')
                    # Integrity check, primarily if text_content might be modified or corrupted independently
                    if row["text_content"] is not None and text_content_decompressed != row["text_content"]:
                        logger.warning(f"Decompressed content mismatch for chunk {chunk_id}. Using decompressed version. Stored: '{row['text_content'][:50]}...', Decompressed: '{text_content_decompressed[:50]}...'")

                    return {
                        "chunk_id": row["chunk_id"],
                        "text_content": text_content_decompressed, # Return decompressed
                        "original_length": row["original_length"],
                        "metadata": json.loads(row["metadata"])
                    }
                return None
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {e}")
            raise
        except zlib.error as e:
            logger.error(f"Failed to decompress chunk {chunk_id}: {e}")
            # Potentially return the raw text_content if stored and valid, or raise
            raise

    def get_all_chunks_for_indexing(self) -> List[Tuple[int, str]]:
        """Fetches all chunks (ID and text content) for building the FAISS index."""
        chunks_for_indexing = []
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                # Fetch either text_content or decompress compressed_content.
                # If text_content is reliably populated and uncompressed, use it.
                # Otherwise, decompress on the fly.
                # Current schema stores text_content, so we use it directly.
                cursor.execute("SELECT chunk_id, text_content, compressed_content FROM chunks ORDER BY chunk_id")
                for row in cursor.fetchall():
                    if row["text_content"] is not None:
                        chunks_for_indexing.append((row["chunk_id"], row["text_content"]))
                    else: # Fallback to decompressing if text_content is missing (should not happen with current add_chunk)
                        try:
                            decompressed_text = zlib.decompress(row["compressed_content"]).decode('utf-8')
                            chunks_for_indexing.append((row["chunk_id"], decompressed_text))
                        except zlib.error as e:
                            logger.error(f"Error decompressing chunk {row['chunk_id']} for indexing: {e}")
                            # Skip this chunk or handle error as appropriate
                return chunks_for_indexing
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve all chunks for indexing: {e}")
            raise
            return []

    def get_total_chunks(self) -> int:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks")
                count = cursor.fetchone()[0]
                return count if count is not None else 0
        except sqlite3.Error as e:
            logger.error(f"Failed to get total chunks count: {e}")
            return 0

    def clear_all_chunks(self):
        """Deletes all chunks from the database. Use with caution."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chunks")
                # Reset autoincrement sequence for chunk_id if desired (SQLite specific)
                # This ensures new IDs start from 1 after a clear.
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")
                conn.commit()
                logger.info("Cleared all chunks from the database.")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear chunks: {e}")
            raise

    def set_info(self, key: str, value: Any):
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)", (key, json.dumps(value)))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to set info for key {key}: {e}")
            raise

    def get_info(self, key: str) -> Optional[Any]:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM info WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row and row["value"] is not None:
                    return json.loads(row["value"])
                return None
        except sqlite3.Error as e:
            logger.error(f"Failed to get info for key {key}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for info key {key}: {e}")
            raise

# Example usage (for testing purposes, can be removed or put in a __main__ block)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for more verbose output during testing
    db_file = "temp_test_memory.db"
    # Ensure a clean slate for testing
    if Path(db_file).exists():
        Path(db_file).unlink()

    db_manager = DatabaseManager(db_file)

    logger.info(f"Initial total chunks: {db_manager.get_total_chunks()}") # Should be 0

    # Add some chunks
    chunk_id1 = db_manager.add_chunk("This is the first text chunk.", {"source": "doc1"})
    chunk_id2 = db_manager.add_chunk("Another piece of text for chunking.", {"source": "doc2", "page": 1})
    chunk_id3 = db_manager.add_chunk("The final chunk in this example.", {"source": "doc1"})
    # Test empty metadata
    chunk_id4 = db_manager.add_chunk("Chunk with no metadata.")


    logger.info(f"Total chunks after adding: {db_manager.get_total_chunks()}") # Should be 4

    # Retrieve a chunk
    retrieved_chunk = db_manager.get_chunk_by_id(chunk_id2)
    if retrieved_chunk:
        logger.info(f"Retrieved chunk ID {retrieved_chunk['chunk_id']}: '{retrieved_chunk['text_content']}' (Metadata: {retrieved_chunk['metadata']})")

    retrieved_chunk_no_meta = db_manager.get_chunk_by_id(chunk_id4)
    if retrieved_chunk_no_meta:
        logger.info(f"Retrieved chunk ID {retrieved_chunk_no_meta['chunk_id']}: '{retrieved_chunk_no_meta['text_content']}' (Metadata: {retrieved_chunk_no_meta['metadata']})")


    # Retrieve all chunks for indexing
    all_chunks_for_indexing = db_manager.get_all_chunks_for_indexing()
    logger.info(f"Chunks for indexing ({len(all_chunks_for_indexing)}):")
    for cid, text in all_chunks_for_indexing:
        logger.info(f"  ID {cid}: {text[:30]}...")

    # Test info table
    db_manager.set_info("last_indexed_timestamp", "2023-10-27T10:00:00Z")
    last_indexed = db_manager.get_info("last_indexed_timestamp")
    logger.info(f"Last indexed: {last_indexed}")

    db_manager.set_info("processed_files", ["file1.txt", "file2.pdf"])
    processed_files = db_manager.get_info("processed_files")
    logger.info(f"Processed files: {processed_files}")

    # Test clearing chunks
    db_manager.clear_all_chunks()
    logger.info(f"Total chunks after clearing: {db_manager.get_total_chunks()}") # Should be 0

    # Add a chunk after clearing to test if auto-increment was reset
    chunk_id_after_clear = db_manager.add_chunk("New chunk after clear.", {"source": "new_doc"})
    logger.info(f"Chunk ID after clear and add: {chunk_id_after_clear}") # Should be 1 if reset worked

    # Clean up the temporary database
    if Path(db_file).exists():
        Path(db_file).unlink()
    logger.info("Cleaned up temporary database.")
