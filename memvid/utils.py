"""
Shared utility functions for Memvid
"""

import io
import json
import qrcode
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import logging
from tqdm import tqdm
import base64
import gzip

from .config import get_default_config, codec_parameters

logger = logging.getLogger(__name__)


def encode_to_qr(data: str) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Encode data to QR code image and extract BitMatrix
    
    Args:
        data: String data to encode
        
    Returns:
        Tuple of (PIL Image of QR code, BitMatrix data dict)
    """

    config = get_default_config()["qr"]
    original_data = data

    if len(data) > 100:
        compressed = gzip.compress(data.encode())
        data = base64.b64encode(compressed).decode()
        data = "GZ:" + data
    
    qr = qrcode.QRCode(
        version=config["version"],
        error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{config['error_correction']}"),
        box_size=config["box_size"],
        border=config["border"],
    )
    
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color=config["fill_color"], back_color=config["back_color"])
    
    bitmatrix_data = extract_qr_bitmatrix(qr)
    if bitmatrix_data:
        bitmatrix_data["data"] = data
        bitmatrix_data["original_data"] = original_data
    
    return img, bitmatrix_data


def decode_qr(image: np.ndarray) -> Optional[str]:
    """
    Decode QR code from image
    
    Args:
        image: OpenCV image array
        
    Returns:
        Decoded string or None if decode fails
    """
    try:
        # Initialize OpenCV QR code detector
        detector = cv2.QRCodeDetector()
        
        # Detect and decode
        data, bbox, straight_qrcode = detector.detectAndDecode(image)
        
        if data:
            # Check if data was compressed
            if data.startswith("GZ:"):
                compressed_data = base64.b64decode(data[3:])
                data = gzip.decompress(compressed_data).decode()
            
            return data
    except Exception as e:
        logger.warning(f"QR decode failed: {e}")
    return None


def qr_to_frame(qr_image: Image.Image, frame_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert QR PIL image to video frame
    
    Args:
        qr_image: PIL Image of QR code
        frame_size: Target frame size (width, height)
        
    Returns:
        OpenCV frame array
    """
    # Resize to fit frame while maintaining aspect ratio
    qr_image = qr_image.resize(frame_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB mode if necessary (handles L, P, etc. modes)
    if qr_image.mode != 'RGB':
        qr_image = qr_image.convert('RGB')
    
    # Convert to numpy array and ensure proper dtype
    img_array = np.array(qr_image, dtype=np.uint8)
    
    # Convert to OpenCV format
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return frame


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract single frame from video
    
    Args:
        video_path: Path to video file
        frame_number: Frame index to extract
        
    Returns:
        OpenCV frame array or None
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            return frame
    finally:
        cap.release()
    return None


@lru_cache(maxsize=1000)
def extract_and_decode_cached(video_path: str, frame_number: int) -> Optional[str]:
    """
    Extract and decode frame with caching
    """
    frame = extract_frame(video_path, frame_number)
    if frame is not None:
        return decode_qr(frame)
    return None


def batch_extract_frames(video_path: str, frame_numbers: List[int], 
                        max_workers: int = 4) -> List[Tuple[int, Optional[np.ndarray]]]:
    """
    Extract multiple frames in parallel
    
    Args:
        video_path: Path to video file
        frame_numbers: List of frame indices
        max_workers: Number of parallel workers
        
    Returns:
        List of (frame_number, frame) tuples
    """
    results = []
    
    # Sort frame numbers for sequential access
    sorted_frames = sorted(frame_numbers)
    
    cap = cv2.VideoCapture(video_path)
    try:
        for frame_num in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            results.append((frame_num, frame if ret else None))
    finally:
        cap.release()
    
    return results


def parallel_decode_qr(frames: List[Tuple[int, np.ndarray]], 
                      max_workers: int = 4) -> List[Tuple[int, Optional[str]]]:
    """
    Decode multiple QR frames in parallel
    
    Args:
        frames: List of (frame_number, frame) tuples
        max_workers: Number of parallel workers
        
    Returns:
        List of (frame_number, decoded_data) tuples
    """
    def decode_frame(item):
        frame_num, frame = item
        if frame is not None:
            data = decode_qr(frame)
            return (frame_num, data)
        return (frame_num, None)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(decode_frame, frames))
    
    return results


def batch_extract_and_decode(video_path: str, frame_numbers: List[int], 
                            max_workers: int = 4, show_progress: bool = False) -> Dict[int, str]:
    """
    Extract and decode multiple frames efficiently
    
    Args:
        video_path: Path to video file
        frame_numbers: List of frame indices
        max_workers: Number of parallel workers
        show_progress: Show progress bar
        
    Returns:
        Dict mapping frame_number to decoded data
    """
    # Extract frames
    frames = batch_extract_frames(video_path, frame_numbers)
    
    # Decode in parallel
    if show_progress:
        frames = tqdm(frames, desc="Decoding QR frames")
    
    decoded = parallel_decode_qr(frames, max_workers)
    
    # Build result dict
    result = {}
    for frame_num, data in decoded:
        if data is not None:
            result[frame_num] = data
    
    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.8:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def save_index(index_data: Dict[str, Any], output_path: str):
    """Save index data to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(index_data, f, indent=2)



def extract_qr_bitmatrix(qr_code) -> Dict[str, Any]:
    """
    Extract BitMatrix data from QR code object
    
    Args:
        qr_code: QRCode object after make() has been called
        
    Returns:
        Dictionary containing BitMatrix data and metadata
    """
    try:
        modules = qr_code.modules
        if modules is None:
            raise ValueError("QR code modules not available - ensure make() was called")
        
        bitmatrix = []
        for row in modules:
            bitmatrix.append([bool(cell) for cell in row])
        
        return {
            "bitmatrix": bitmatrix,
            "version": qr_code.version,
            "error_correction": qr_code.error_correction,
            "box_size": qr_code.box_size,
            "border": qr_code.border
        }
    except Exception as e:
        logger.warning(f"BitMatrix extraction failed: {e}")
        return None


def decode_qr_from_bitmatrix(bitmatrix_data: Dict[str, Any]) -> Optional[str]:
    """
    Decode QR data directly from BitMatrix JSON data
    
    Args:
        bitmatrix_data: Dictionary containing BitMatrix and metadata
        
    Returns:
        Decoded string or None if decode fails
    """
    try:
        if "data" in bitmatrix_data:
            data = bitmatrix_data["data"]
            
            if data.startswith("GZ:"):
                compressed_data = base64.b64decode(data[3:])
                data = gzip.decompress(compressed_data).decode()
            
            return data
            
    except Exception as e:
        logger.warning(f"BitMatrix decode failed: {e}")
    return None


def extract_and_decode_from_json(json_path: str) -> Optional[str]:
    """
    Extract and decode data directly from JSON file
    
    Args:
        json_path: Path to JSON file containing BitMatrix data
        
    Returns:
        Decoded string or None if decode fails
    """
    try:
        with open(json_path, 'r') as f:
            bitmatrix_data = json.load(f)
        
        if "original_data" in bitmatrix_data:
            return bitmatrix_data["original_data"]
        elif "data" in bitmatrix_data:
            data = bitmatrix_data["data"]
            if data.startswith("GZ:"):
                compressed_data = base64.b64decode(data[3:])
                data = gzip.decompress(compressed_data).decode()
            return data
            
    except Exception as e:
        logger.warning(f"JSON decode failed: {e}")
    return None


def batch_extract_and_decode_json(frames_dir: str, frame_numbers: List[int]) -> Dict[int, str]:
    """
    Extract and decode multiple frames from JSON files efficiently
    
    Args:
        frames_dir: Directory containing JSON files
        frame_numbers: List of frame numbers to decode
        
    Returns:
        Dict mapping frame_number to decoded data
    """
    from pathlib import Path
    results = {}
    
    for frame_num in frame_numbers:
        json_path = Path(frames_dir) / f"frame_{frame_num:06d}.json"
        if json_path.exists():
            decoded = extract_and_decode_from_json(str(json_path))
            if decoded:
                results[frame_num] = decoded
    
    return results


def load_index(index_path: str) -> Dict[str, Any]:
    """Load index data from JSON file"""
    with open(index_path, 'r') as f:
        return json.load(f)
