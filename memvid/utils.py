"""
Shared utility functions for Memvid
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    Tries to break at sentence boundaries for cleaner cuts.
    
    Args:
        text: Text to chunk.
        chunk_size: Target chunk size in characters.
        overlap: Overlap between chunks.
        
    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    text_len = len(text)

    if text_len == 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        logger.warning(f"Overlap ({overlap}) is greater than or equal to chunk_size ({chunk_size}). This may lead to inefficient chunking or errors.")
        # Fallback to non-overlapping if overlap is problematic, or raise error
        # For now, let it proceed but it might produce many identical chunks or behave unexpectedly.

    while start < text_len:
        end = start + chunk_size
        
        current_segment_text = text[start:end] # This is the potential chunk up to chunk_size
        
        actual_end = end
        
        if end < text_len: # If we are not at the end of the text, try to find a better break point
            # Search for sentence terminators (. ! ?) in the latter part of the segment
            # Prefer breaks that are not too early in the chunk.
            # Search from the end of `current_segment_text` backwards.
            
            best_break_offset = -1

            for i in range(len(current_segment_text) - 1, 0, -1):
                char = current_segment_text[i]
                if char in ['.', '!', '?']:
                    # Check if this punctuation is followed by a space or is at the end of the segment
                    is_sentence_boundary = (i == len(current_segment_text) - 1) or \
                                           (i + 1 < len(current_segment_text) and current_segment_text[i+1].isspace())

                    if is_sentence_boundary:
                        # Avoid breaking in numbers like "3.14" or common abbreviations like "U.S."
                        # This is a simple heuristic. More robust sentence tokenization is complex.
                        if char == '.':
                            if i > 0 and current_segment_text[i-1].isdigit() and \
                               (i + 1 < len(current_segment_text) and current_segment_text[i+1].isdigit()):
                                continue # Likely a number, not a sentence end
                            if i > 0 and current_segment_text[i-1].isupper() and \
                               (i + 1 < len(current_segment_text) and current_segment_text[i+1].isalpha()): # e.g. U.S. President
                                # This is tricky, could be an initial. For simplicity, we might break.
                                # Or decide not to break on single letter followed by period then letter.
                                pass # Allow break for now

                        # Found a potential break point. We want the one closest to original `end`.
                        # Since we iterate backwards, the first one found is the latest one.
                        best_break_offset = i
                        break

            if best_break_offset != -1:
                # Check if using this break point is "reasonable"
                # e.g., the chunk should not be too small (e.g., less than overlap size)
                if (start + best_break_offset + 1) > (start + overlap * 0.5): # Ensure chunk is not excessively small
                    actual_end = start + best_break_offset + 1
                # else, we stick with the original 'end' based on chunk_size
        
        final_chunk_text = text[start:actual_end]
        stripped_chunk = final_chunk_text.strip()
        
        if stripped_chunk: # Only add non-empty chunks
            chunks.append(stripped_chunk)
        
        next_start = actual_end - overlap
        
        if next_start <= start : # Ensure progress is made
            if actual_end >= text_len : # If we've processed till the end
                 break
            next_start = start + 1 # Force move forward by at least one char if no other progress
            if next_start > actual_end : # If overlap is too large, just move to end of current chunk
                 next_start = actual_end


        start = next_start
        if start >= text_len: # Condition to ensure we don't loop if next_start calculation is off
            break

    return chunks
