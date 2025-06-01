"""
MemvidRetriever - Fast semantic search, QR frame extraction, and context assembly
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional # Removed Tuple as search result format changed
import time
from functools import lru_cache # Ensure this is present and used

# Removed cv2 and concurrent.futures related imports
# Removed QR/frame specific utils
from .index import IndexManager
from .config import get_default_config
from .database import DatabaseManager # Added DatabaseManager

logger = logging.getLogger(__name__)


class MemvidRetriever:
    """Retrieves text chunks from a database using semantic search."""
    
    def __init__(self, index_file_path_prefix: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize MemvidRetriever.
        
        Args:
            index_file_path_prefix: Path prefix for the FAISS index files.
            config: Optional configuration dictionary.
        """
        self.index_file_path_prefix = index_file_path_prefix
        self.config = config or get_default_config()
        
        # Initialize IndexManager
        self.index_manager = IndexManager(self.config)
        self.index_manager.load(index_file_path_prefix)
        
        # Initialize DatabaseManager
        db_config = self.config.get("database", {})
        db_path = db_config.get("path")
        self.db_manager = DatabaseManager(db_path=db_path)
        
        # Caching for DB chunk details is handled by _get_db_chunk_details method's lru_cache
        
        logger.info(f"Initialized retriever. Index: {index_file_path_prefix}, DB: {self.db_manager.db_path}")

    @lru_cache(maxsize=1000) # Default cache size, consider making configurable via self.config
    def _get_db_chunk_details(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Fetches chunk details (text & metadata) from DB. Cached."""
        try:
            return self.db_manager.get_chunk_by_id(chunk_id)
        except Exception as e:
            logger.error(f"Error fetching chunk {chunk_id} from DB: {e}")
            return None

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search for relevant chunks using semantic search.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of relevant text chunks.
        """
        start_time = time.time()
        
        # search_results_ids is List[(chunk_id: int, distance: float)]
        search_results_ids = self.index_manager.search(query, top_k)
        
        results_text = []
        for chunk_id, distance in search_results_ids:
            chunk_details = self._get_db_chunk_details(chunk_id)
            if chunk_details and "text_content" in chunk_details:
                results_text.append(chunk_details["text_content"])
            elif chunk_details:
                logger.warning(f"Chunk {chunk_id} details fetched but missing 'text_content'.")
            else:
                logger.warning(f"Could not retrieve details for chunk_id {chunk_id} from database.")

        elapsed = time.time() - start_time
        logger.info(f"Search for '{query[:50]}...' (top {top_k}) completed in {elapsed:.3f}s, found {len(results_text)} results.")
        return results_text
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[str]:
        """
        Get specific chunk text by ID from database.
        
        Args:
            chunk_id: Chunk ID.
            
        Returns:
            Chunk text or None if not found or error.
        """
        chunk_details = self._get_db_chunk_details(chunk_id)
        return chunk_details["text_content"] if chunk_details else None

    def search_with_metadata(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks and return them with their metadata and scores.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of result dictionaries, each containing "text", "score", "chunk_id", and "metadata".
        """
        start_time = time.time()
        search_results_ids = self.index_manager.search(query, top_k) # List[(id, dist)]
        
        results = []
        for chunk_id, distance in search_results_ids:
            chunk_details = self._get_db_chunk_details(chunk_id)
            if chunk_details:
                results.append({
                    "text": chunk_details["text_content"],
                    "score": 1.0 / (1.0 + distance) if distance >= 0 else 0,
                    "chunk_id": chunk_id,
                    "metadata": chunk_details.get("metadata", {})
                })
            else:
                logger.warning(f"Could not retrieve details for chunk_id {chunk_id} in search_with_metadata.")

        elapsed = time.time() - start_time
        logger.info(f"Search_with_metadata for '{query[:50]}...' (top {top_k}) completed in {elapsed:.3f}s, found {len(results)} results.")
        return results
    
    def get_context_window(self, chunk_id: int, window_size: int = 2) -> List[str]:
        """
        Get a central chunk and its surrounding context from the database.
        
        Args:
            chunk_id: ID of the central chunk.
            window_size: Number of chunks to retrieve before and after the central chunk.
            
        Returns:
            List of chunk texts, including the central chunk and its context.
        """
        context_chunks = []
        # Iterate from -window_size to +window_size relative to the target chunk_id
        # Note: Assumes chunk_ids are somewhat contiguous for context, which might not always be true.
        # If chunk_ids can be arbitrary, this method might need rethinking or relying on metadata (e.g., sequence numbers).
        # For now, it assumes that `chunk_id + offset` is a meaningful way to get neighbors.
        for offset in range(-window_size, window_size + 1):
            current_chunk_id = chunk_id + offset
            # We could add a check here: if offset is 0, ensure it's the actual target chunk.
            # However, get_chunk_by_id will return None if the ID doesn't exist.
            chunk_text = self.get_chunk_by_id(current_chunk_id)
            if chunk_text:
                context_chunks.append(chunk_text)

        return context_chunks
    
    def clear_cache(self):
        """Clear retriever's chunk cache."""
        if hasattr(self._get_db_chunk_details, 'cache_clear'):
            self._get_db_chunk_details.cache_clear()
            logger.info("Cleared retriever's chunk cache.")
        else:
            logger.warning("Retriever's _get_db_chunk_details method does not have cache_clear().")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        cache_info_dict = {}
        if hasattr(self._get_db_chunk_details, 'cache_info'):
            ci = self._get_db_chunk_details.cache_info()
            cache_info_dict = {
                "hits": ci.hits, "misses": ci.misses,
                "maxsize": ci.maxsize, "currsize": ci.currsize
            }
        else:
            cache_info_dict = {"info": "Cache info not available for _get_db_chunk_details."}

        return {
            "index_file_prefix": self.index_file_path_prefix,
            "db_path": str(self.db_manager.db_path),
            "cache_stats": cache_info_dict,
            "index_stats": self.index_manager.get_stats()
        }