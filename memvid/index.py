"""
Index management for embeddings and vector search
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
# import pickle # Removed as it seems unused
from tqdm import tqdm # Keep for potential future use, or remove if show_progress_bar in encode is enough

from .config import get_default_config
from .database import DatabaseManager # Added DatabaseManager

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages embeddings, FAISS index, and metadata for fast retrieval"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IndexManager
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_default_config()
        self.embedding_model = SentenceTransformer(self.config["embedding"]["model"])
        self.dimension = self.config["embedding"]["dimension"]
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Metadata storage is now handled by DatabaseManager and the FAISS index itself (IDs)
        
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        index_type = self.config["index"]["type"]
        
        if index_type == "Flat":
            # Exact search - best quality, slower for large datasets
            index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVF":
            nlist = self.config["index"].get("nlist", 100) # Default nlist if not in config
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        # Add ID mapping for retrieval. IndexIDMap wraps the actual index.
        # So self.index will be IndexIDMap, and self.index.index will be the IVF or Flat index.
        index = faiss.IndexIDMap(index)
        return index

    def clear_index(self):
        """Resets the FAISS index to an empty state."""
        if self.index:
            self.index.reset() # Clears all vectors from the index
        # Re-initialize the index to ensure it's clean, especially if it was trained (e.g. IVF)
        self.index = self._create_index()
        logger.info("FAISS index has been cleared and re-initialized.")

    def build_index_from_db(self, db_manager: DatabaseManager, show_progress: bool = True):
        """Builds or rebuilds the FAISS index using chunks from the DatabaseManager."""
        self.clear_index() # Start with a fresh index

        logger.info("Fetching chunks from database for indexing...")
        db_chunks = db_manager.get_all_chunks_for_indexing()

        if not db_chunks:
            logger.warning("No chunks found in the database to index.")
            return

        chunk_ids_from_db = [item[0] for item in db_chunks]
        text_contents = [item[1] for item in db_chunks]

        logger.info(f"Generating embeddings for {len(text_contents)} chunks...")
        embeddings = self.embedding_model.encode(
            text_contents,
            show_progress_bar=show_progress,
            batch_size=self.config.get("retrieval", {}).get("batch_size", 32)
        )
        embeddings_np = np.array(embeddings).astype('float32')
        chunk_ids_np = np.array(chunk_ids_from_db, dtype=np.int64)

        # Train index if needed (for IVF type)
        # self.index is IndexIDMap, self.index.index is the actual IndexIVFFlat or IndexFlatL2
        actual_index = self.index.index
        if isinstance(actual_index, faiss.IndexIVFFlat) and not actual_index.is_trained:
            logger.info(f"Training FAISS index (type: IVF, nlist: {actual_index.nlist})...")
            actual_index.train(embeddings_np)
            logger.info(f"FAISS index trained. Trained points: {actual_index.ntotal if hasattr(actual_index, 'ntotal') else 'N/A'}")

        logger.info(f"Adding {embeddings_np.shape[0]} vectors to FAISS index...")
        self.index.add_with_ids(embeddings_np, chunk_ids_np)

        logger.info(f"Successfully built FAISS index. Total indexed chunks: {self.index.ntotal}")
        
        # Store total indexed chunks in DB info table
        db_manager.set_info("total_indexed_chunks", self.index.ntotal)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar chunks.
        Returns:
            List of (chunk_id, distance) tuples, where chunk_id is the ID from the database.
        """
        query_embedding = self.embedding_model.encode([query])
        query_embedding_np = np.array(query_embedding).astype('float32')

        if self.index.ntotal == 0:
            logger.warning("Search attempted on an empty index.")
            return []

        distances, chunk_ids_from_faiss = self.index.search(query_embedding_np, top_k)

        results = []
        for i in range(chunk_ids_from_faiss.shape[1]):
            chunk_id = int(chunk_ids_from_faiss[0, i])
            dist = float(distances[0, i])
            if chunk_id != -1: # FAISS uses -1 for no result or padding
                results.append((chunk_id, dist))
        return results
    
    def save(self, path_prefix: str):
        """
        Save FAISS index and essential metadata to disk.
        Args:
            path_prefix: Path prefix to save index and metadata (e.g., "my_index")
                         Results in "my_index.faiss" and "my_index.indexinfo.json".
        """
        path = Path(path_prefix)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss_index_file = str(path.with_suffix('.faiss'))
        faiss.write_index(self.index, faiss_index_file)
        logger.info(f"Saved FAISS index to {faiss_index_file}")

        index_info = {
            "config": {
                "embedding_model": self.config.get("embedding", {}).get("model"),
                "embedding_dimension": self.dimension,
                "index_type": self.config.get("index", {}).get("type"),
                "nlist_for_ivf": self.config.get("index", {}).get("nlist") if self.config.get("index", {}).get("type") == "IVF" else None,
            },
            "total_indexed_chunks": self.index.ntotal
        }
        
        info_file = str(path.with_suffix('.indexinfo.json'))
        with open(info_file, 'w') as f:
            json.dump(index_info, f, indent=2)
        logger.info(f"Saved index info to {info_file}")
    
    def load(self, path_prefix: str):
        """
        Load FAISS index and essential metadata from disk.
        Args:
            path_prefix: Path prefix to load index and metadata from.
        """
        path = Path(path_prefix)
        faiss_index_file = str(path.with_suffix('.faiss'))
        
        if not Path(faiss_index_file).exists():
            logger.warning(f"FAISS index file not found: {faiss_index_file}. Initializing a new empty index.")
            self.index = self._create_index()
        else:
            self.index = faiss.read_index(faiss_index_file)
            logger.info(f"Loaded FAISS index from {faiss_index_file}. Total items: {self.index.ntotal}")

        info_file = str(path.with_suffix('.indexinfo.json'))
        if Path(info_file).exists():
            with open(info_file, 'r') as f:
                index_info = json.load(f)

            loaded_config_info = index_info.get("config", {})
            logger.info(f"Loaded index info: {loaded_config_info}")

            current_embedding_model = self.config.get("embedding", {}).get("model")
            loaded_embedding_model = loaded_config_info.get("embedding_model")
            if current_embedding_model != loaded_embedding_model:
                logger.warning(
                    f"Mismatch in embedding model between current config ('{current_embedding_model}') "
                    f"and loaded index info ('{loaded_embedding_model}'). Using current config's model."
                )
            # Further checks can be added for dimension, index_type etc. if critical for compatibility
        else:
            logger.warning(f"Index info file not found: {info_file}. Index may not be optimally configured if parameters changed since last save.")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        actual_index = self.index.index if self.index else None
        is_trained_status = "N/A (No index or not applicable)"
        if actual_index:
            if isinstance(actual_index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ, faiss.IndexIVFScalarQuantizer)): # Add other IVF types if used
                 is_trained_status = actual_index.is_trained
            elif isinstance(actual_index, faiss.IndexFlatL2): # Flat indexes don't require training
                 is_trained_status = "N/A (Flat index)"

        return {
            "total_indexed_chunks": self.index.ntotal if self.index else 0,
            "index_type": self.config.get("index", {}).get("type"),
            "embedding_model": self.config.get("embedding", {}).get("model"),
            "dimension": self.dimension,
            "is_trained": is_trained_status,
        }