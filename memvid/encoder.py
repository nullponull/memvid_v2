"""
MemvidEncoder - Handles chunking and QR video creation
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import numpy as np # Kept for get_stats, though its usage might change

from .utils import chunk_text # Removed QR/video utils
from .index import IndexManager
from .config import get_default_config
from .database import DatabaseManager # Added DatabaseManager

logger = logging.getLogger(__name__)


class MemvidEncoder:
    """Stores text chunks in a database and builds a searchable index."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MemvidEncoder
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_default_config()
        db_config = self.config.get("database", {})
        db_path = db_config.get("path") # Uses DEFAULT_DATABASE_PATH from config if not overridden
        self.db_manager = DatabaseManager(db_path=db_path)
        self.index_manager = IndexManager(self.config)
        
    def add_chunks(self, chunks: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None):
        """
        Add text chunks to the database.
        
        Args:
            chunks: List of text chunks.
            metadata_list: Optional list of metadata dictionaries, one per chunk.
        """
        if metadata_list and len(metadata_list) != len(chunks):
            raise ValueError("If metadata_list is provided, it must have the same length as chunks.")

        num_added = 0
        for i, chunk_text_content in enumerate(chunks):
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            try:
                self.db_manager.add_chunk(chunk_text_content, meta)
                num_added += 1
            except Exception as e:
                logger.error(f"Failed to add chunk to DB: {chunk_text_content[:50]}... Error: {e}")
        logger.info(f"Attempted to add {len(chunks)} chunks to DB. Successfully added: {num_added}")

    def add_text(self, text: str, chunk_size: int = 500, overlap: int = 50, metadata: Optional[Dict[str, Any]] = None):
        """
        Add text, automatically chunk it, and store it in the database.
        
        Args:
            text: Text to chunk and add.
            chunk_size: Target chunk size.
            overlap: Overlap between chunks.
            metadata: Optional metadata dictionary to associate with all created chunks.
        """
        new_chunks = chunk_text(text, chunk_size, overlap)
        metadata_list_for_chunks = [metadata.copy() if metadata else {} for _ in new_chunks]
        self.add_chunks(new_chunks, metadata_list_for_chunks)
    
    def add_pdf(self, pdf_path: str, chunk_size: int = 800, overlap: int = 100):
        """
        Extract text from PDF, add as chunks to the database with metadata.
        
        Args:
            pdf_path: Path to PDF file.
            chunk_size: Target chunk size.
            overlap: Overlap between chunks.
        """
        try:
            import PyPDF2
        except ImportError:
            logger.error("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
            raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
        
        if not Path(pdf_path).exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = ""
        num_pages = 0
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                logger.info(f"Extracting text from {num_pages} pages of {Path(pdf_path).name}")

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text: # Ensure text was extracted
                        text += page_text + "\n\n" # Add double newline to mark page breaks more clearly if needed
            
            if text.strip():
                pdf_metadata = {"source_pdf": Path(pdf_path).name, "pdf_total_pages": num_pages}
                self.add_text(text, chunk_size, overlap, metadata=pdf_metadata)
                logger.info(f"Added PDF content: {len(text)} characters from {Path(pdf_path).name}")
            else:
                logger.warning(f"No text extracted from PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            # Potentially re-raise or handle as per application needs
            raise

    def build_memory(self, index_file_path_prefix: str, show_progress: bool = True) -> Dict[str, Any]:
        """
        Ensures all added chunks are stored and builds/updates the search index.
        
        Args:
            index_file_path_prefix: Prefix for the output index file (e.g., "my_memory_index").
                                     Actual files will be like "my_memory_index.faiss" and "my_memory_index_meta.json".
            show_progress: Show progress bar for indexing.
            
        Returns:
            Dictionary with build statistics.
        """
        logger.info("Building memory: Ensuring all chunks are stored and indexing...")

        # Chunks are already added to DB via add_chunk/add_text/add_pdf.
        # Indexing (this method will be properly defined in IndexManager in the next plan step):
        if hasattr(self.index_manager, 'build_index_from_db'):
            self.index_manager.build_index_from_db(self.db_manager, show_progress=show_progress)
        else:
            logger.warning("IndexManager does not yet have 'build_index_from_db'. Indexing skipped in MemvidEncoder.")

        self.index_manager.save(index_file_path_prefix) # Saves the FAISS index + its metadata
        
        db_stats = {
            "db_file": str(self.db_manager.db_path),
            "db_size_mb": self.db_manager.db_path.stat().st_size / (1024 * 1024) if self.db_manager.db_path.exists() else 0,
        }
        
        stats = {
            "total_chunks_in_db": self.db_manager.get_total_chunks(),
            "index_file_prefix": index_file_path_prefix,
            "index_stats": self.index_manager.get_stats(), # IndexManager.get_stats() will also need update
            "database_stats": db_stats
        }
        logger.info(f"Successfully built memory. Index saved with prefix: {index_file_path_prefix}")
        return stats

    def clear(self):
        """Clear all chunks from the database and reset the index."""
        self.db_manager.clear_all_chunks()
        if hasattr(self.index_manager, 'clear_index'):
            self.index_manager.clear_index()
        else:
            logger.warning("IndexManager does not yet have 'clear_index'. Re-initializing IndexManager.")
            self.index_manager = IndexManager(self.config) # Fallback
        logger.info("Cleared all chunks from database and reset index manager.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoder and database statistics"""
        return {
            "total_chunks_in_db": self.db_manager.get_total_chunks(),
            # "total_characters_in_db": "N/A (requires iterating DB or pre-calculation)",
            # "avg_chunk_size_in_db": "N/A (requires iterating DB or pre-calculation)",
            "config": self.config,
            "db_path": str(self.db_manager.db_path)
        }
    
    @classmethod
    def from_file(cls, file_path: str, chunk_size: int = 500, 
                  overlap: int = 50, config: Optional[Dict[str, Any]] = None) -> 'MemvidEncoder':
        """
        Create encoder from text file, adding content to the database.
        
        Args:
            file_path: Path to text file.
            chunk_size: Target chunk size.
            overlap: Overlap between chunks.
            config: Optional configuration.
            
        Returns:
            MemvidEncoder instance with content loaded.
        """
        encoder = cls(config)
        file_p = Path(file_path)
        if not file_p.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_p, 'r', encoding='utf-8') as f:
                text = f.read()

            if text.strip():
                file_metadata = {"source_file": file_p.name, "source_path": str(file_p.resolve())}
                encoder.add_text(text, chunk_size, overlap, metadata=file_metadata)
                logger.info(f"Successfully processed text file: {file_path}")
            else:
                logger.warning(f"File is empty or contains only whitespace: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
        return encoder
    
    @classmethod
    def from_documents(cls, documents: List[str], chunk_size: int = 500,
                      overlap: int = 50, config: Optional[Dict[str, Any]] = None,
                      metadata_list: Optional[List[Dict[str,Any]]] = None) -> 'MemvidEncoder':
        """
        Create encoder from a list of document strings, adding content to the database.
        
        Args:
            documents: List of document strings.
            chunk_size: Target chunk size.
            overlap: Overlap between chunks.
            config: Optional configuration.
            metadata_list: Optional list of metadata dictionaries, one for each document.
                           If provided, metadata will be associated with chunks from the respective document.
            
        Returns:
            MemvidEncoder instance with content loaded.
        """
        encoder = cls(config)
        
        if metadata_list and len(metadata_list) != len(documents):
            raise ValueError("If metadata_list is provided, it must have the same length as documents.")

        for i, doc_text in enumerate(documents):
            doc_metadata = {}
            if metadata_list and i < len(metadata_list):
                doc_metadata = metadata_list[i].copy() # Use a copy to avoid modification issues

            # Add a default source identifier if not present in provided metadata
            if "source" not in doc_metadata:
                 doc_metadata["source"] = f"document_index_{i}"

            if doc_text.strip():
                encoder.add_text(doc_text, chunk_size, overlap, metadata=doc_metadata)
            else:
                logger.warning(f"Document at index {i} is empty or contains only whitespace. Skipping.")
        
        return encoder