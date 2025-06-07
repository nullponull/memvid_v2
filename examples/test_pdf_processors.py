#!/usr/bin/env python3
"""
Test script for different PDF processors in Memvid
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memvid import MemvidEncoder


def test_pdf_processor(pdf_path: str, processor: str) -> dict:
    """Test a specific PDF processor and return results"""
    print(f"\n📄 Testing {processor.upper()} processor...")
    
    try:
        encoder = MemvidEncoder()
        start_time = time.time()
        
        encoder.add_pdf(pdf_path, pdf_processor=processor)
        
        processing_time = time.time() - start_time
        
        if encoder.chunks:
            total_chars = sum(len(chunk) for chunk in encoder.chunks)
            avg_chunk_size = total_chars / len(encoder.chunks)
            
            result = {
                'success': True,
                'processor': processor,
                'chunks': len(encoder.chunks),
                'total_chars': total_chars,
                'avg_chunk_size': avg_chunk_size,
                'processing_time': processing_time,
                'sample_text': encoder.chunks[0][:200] + "..." if encoder.chunks else ""
            }
            
            print(f"✅ Success: {len(encoder.chunks)} chunks, {total_chars} chars")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Sample: {result['sample_text']}")
            
            return result
        else:
            print(f"❌ No text extracted")
            return {
                'success': False,
                'processor': processor,
                'error': 'No text extracted',
                'processing_time': processing_time
            }
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'success': False,
            'processor': processor,
            'error': str(e),
            'processing_time': 0
        }


def check_dependencies():
    """Check which PDF processors are available"""
    processors = {
        'pypdf2': False,
        'pymupdf': False,
        'ocr_tesseract': False,
        'ocr_easyocr': False,
        'ocr_handwritten': False
    }
    
    print("🔍 Checking PDF processor dependencies...")
    
    # Check PyPDF2
    try:
        import PyPDF2
        processors['pypdf2'] = True
        print("✅ PyPDF2: Available")
    except ImportError:
        print("❌ PyPDF2: Not installed (pip install PyPDF2)")
    
    # Check PyMuPDF
    try:
        import fitz
        processors['pymupdf'] = True
        print("✅ PyMuPDF: Available")
    except ImportError:
        print("❌ PyMuPDF: Not installed (pip install pymupdf)")
    
    # Check Tesseract OCR
    try:
        import pytesseract
        import fitz
        from PIL import Image
        processors['ocr_tesseract'] = True
        print("✅ Tesseract OCR: Available")
    except ImportError:
        print("❌ Tesseract OCR: Missing dependencies (pip install pymupdf pytesseract Pillow)")
    
    # Check EasyOCR
    try:
        import easyocr
        import fitz
        import numpy as np
        from PIL import Image
        processors['ocr_easyocr'] = True
        print("✅ EasyOCR: Available")
    except ImportError:
        print("❌ EasyOCR: Missing dependencies (pip install pymupdf easyocr numpy Pillow)")
    
    # Check Specialized Handwritten OCR
    try:
        import easyocr
        import pytesseract
        import fitz
        import cv2
        import numpy as np
        from PIL import Image
        processors['ocr_handwritten'] = True
        print("✅ Specialized Handwritten OCR: Available")
    except ImportError:
        print("❌ Specialized Handwritten OCR: Missing dependencies (pip install pymupdf easyocr pytesseract opencv-python numpy Pillow)")
    
    return processors


def compare_processors(pdf_path: str, processors_to_test: list = None):
    """Compare different PDF processors on the same file"""
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
    print(f"📁 Testing PDF: {Path(pdf_path).name} ({file_size:.1f} MB)")
    
    # Check available processors
    available = check_dependencies()
    
    if processors_to_test is None:
        processors_to_test = [p for p, avail in available.items() if avail]
    
    if not processors_to_test:
        print("❌ No PDF processors available. Install dependencies first.")
        return
    
    print(f"\n🧪 Testing {len(processors_to_test)} processors: {', '.join(processors_to_test)}")
    
    results = []
    
    for processor in processors_to_test:
        if not available.get(processor, False):
            print(f"\n⚠️  Skipping {processor} - dependencies not available")
            continue
            
        result = test_pdf_processor(pdf_path, processor)
        results.append(result)
    
    # Print comparison table
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Processor':<15} {'Status':<10} {'Chunks':<8} {'Chars':<10} {'Time(s)':<8} {'Chars/s':<10}")
    print("-" * 80)
    
    for result in results:
        if result['success']:
            chars_per_sec = result['total_chars'] / result['processing_time'] if result['processing_time'] > 0 else 0
            print(f"{result['processor']:<15} {'✅ OK':<10} {result['chunks']:<8} {result['total_chars']:<10} {result['processing_time']:<8.2f} {chars_per_sec:<10.0f}")
        else:
            print(f"{result['processor']:<15} {'❌ FAIL':<10} {'-':<8} {'-':<10} {'-':<8} {'-':<10}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # Find best by different criteria
        most_text = max(successful_results, key=lambda x: x['total_chars'])
        fastest = min(successful_results, key=lambda x: x['processing_time'])
        
        print(f"📈 Most text extracted: {most_text['processor']} ({most_text['total_chars']} chars)")
        print(f"⚡ Fastest processing: {fastest['processor']} ({fastest['processing_time']:.2f}s)")
        
        if any('ocr' in r['processor'] for r in successful_results):
            print(f"🔍 For scanned/handwritten PDFs: Use ocr_tesseract or ocr_easyocr")
        
        print(f"📄 For digital PDFs: Use pymupdf (better) or pypdf2 (standard)")
    else:
        print("❌ No processors worked successfully")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test and compare different PDF processors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_pdf_processors.py sample.pdf
  python test_pdf_processors.py scanned.pdf --processors ocr_tesseract ocr_easyocr
  python test_pdf_processors.py document.pdf --processors pypdf2 pymupdf
        """
    )
    
    parser.add_argument('pdf_path', help='Path to PDF file to test')
    parser.add_argument(
        '--processors', 
        nargs='+',
        choices=['pypdf2', 'pymupdf', 'ocr_tesseract', 'ocr_easyocr', 'ocr_handwritten'],
        help='Specific processors to test (default: all available)'
    )
    
    args = parser.parse_args()
    
    print("🧪 PDF Processor Testing Tool")
    print("=" * 50)
    
    compare_processors(args.pdf_path, args.processors)
    
    print(f"\n🎯 Usage examples:")
    print(f"# Use PyMuPDF for better digital PDF extraction:")
    print(f"python file_chat.py --files {args.pdf_path} --pdf-processor pymupdf")
    print(f"")
    print(f"# Use OCR for scanned/handwritten PDFs:")
    print(f"python file_chat.py --files {args.pdf_path} --pdf-processor ocr_tesseract")
    print(f"python file_chat.py --files {args.pdf_path} --pdf-processor ocr_easyocr")


if __name__ == "__main__":
    main() 