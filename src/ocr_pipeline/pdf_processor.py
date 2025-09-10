from typing import List, Dict, Optional, Union
import logging
import os
import io
from pathlib import Path
import tempfile

from PIL import Image
import fitz  # PyMuPDF
import numpy as np


class PDFProcessor:
    """Enhanced PDF processing with page extraction and optimization for OCR."""
    
    def __init__(self, dpi: int = 300, output_format: str = "PNG"):
        """
        Initialize PDF processor.
        
        Args:
            dpi: Resolution for PDF to image conversion
            output_format: Output image format (PNG, JPEG, TIFF)
        """
        self.dpi = dpi
        self.output_format = output_format.upper()
        
    def extract_pages_as_images(self, pdf_path: str, page_range: Optional[tuple] = None) -> List[Image.Image]:
        """
        Extract PDF pages as PIL Images.
        
        Args:
            pdf_path: Path to PDF file
            page_range: Optional tuple (start, end) for page range, 0-indexed
            
        Returns:
            List of PIL Images, one per page
        """
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Determine page range
            if page_range:
                start_page, end_page = page_range
                start_page = max(0, start_page)
                end_page = min(total_pages, end_page)
            else:
                start_page, end_page = 0, total_pages
            
            logging.info(f"Extracting pages {start_page}-{end_page-1} from {total_pages} total pages")
            
            for page_num in range(start_page, end_page):
                page = doc.load_page(page_num)
                
                # Convert to image with specified DPI
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # 72 is default PDF DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                images.append(img)
                logging.debug(f"Extracted page {page_num + 1}: {img.size}")
            
            doc.close()
            logging.info(f"Successfully extracted {len(images)} pages")
            
        except Exception as e:
            logging.error(f"Error extracting PDF pages: {str(e)}")
            raise
        
        return images
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict:
        """Extract PDF metadata and document information."""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            info = {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'total_pages': len(doc),
                'file_size': os.path.getsize(pdf_path),
                'encrypted': doc.needs_pass,
                'pdf_version': f"{doc.pdf_version()[0]}.{doc.pdf_version()[1]}"
            }
            
            # Get page dimensions for first page
            if len(doc) > 0:
                first_page = doc.load_page(0)
                rect = first_page.rect
                info['page_width'] = rect.width
                info['page_height'] = rect.height
                info['page_dimensions'] = f"{rect.width:.1f} x {rect.height:.1f} points"
            
            doc.close()
            return info
            
        except Exception as e:
            logging.error(f"Error extracting PDF metadata: {str(e)}")
            return {}
    
    def extract_text_from_pdf(self, pdf_path: str, page_range: Optional[tuple] = None) -> List[Dict]:
        """
        Extract existing text from PDF (if available) before OCR.
        
        Args:
            pdf_path: Path to PDF file
            page_range: Optional tuple (start, end) for page range
            
        Returns:
            List of dictionaries with page text and metadata
        """
        text_data = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Determine page range
            if page_range:
                start_page, end_page = page_range
                start_page = max(0, start_page)
                end_page = min(total_pages, end_page)
            else:
                start_page, end_page = 0, total_pages
            
            for page_num in range(start_page, end_page):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                
                # Get text blocks with positioning
                blocks = page.get_text("dict")
                
                # Count text elements
                text_blocks = []
                for block in blocks.get("blocks", []):
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["text"].strip():
                                    text_blocks.append({
                                        'text': span["text"],
                                        'bbox': span["bbox"],
                                        'font': span["font"],
                                        'size': span["size"]
                                    })
                
                page_data = {
                    'page_number': page_num + 1,
                    'text': text.strip(),
                    'text_length': len(text.strip()),
                    'has_text': bool(text.strip()),
                    'text_blocks': text_blocks,
                    'block_count': len(text_blocks)
                }
                
                text_data.append(page_data)
            
            doc.close()
            logging.info(f"Extracted text from {len(text_data)} pages")
            
        except Exception as e:
            logging.error(f"Error extracting PDF text: {str(e)}")
            raise
        
        return text_data
    
    def is_scanned_pdf(self, pdf_path: str, text_threshold: float = 0.1) -> bool:
        """
        Determine if PDF is likely a scanned document (needs OCR).
        
        Args:
            pdf_path: Path to PDF file
            text_threshold: Minimum ratio of pages with text to consider as text-based PDF
            
        Returns:
            True if PDF appears to be scanned (low text content)
        """
        try:
            text_data = self.extract_text_from_pdf(pdf_path)
            
            if not text_data:
                return True  # No pages processed, assume scanned
            
            pages_with_text = sum(1 for page in text_data if page['has_text'])
            text_ratio = pages_with_text / len(text_data)
            
            logging.info(f"PDF text analysis: {pages_with_text}/{len(text_data)} pages have text (ratio: {text_ratio:.2f})")
            
            return text_ratio < text_threshold
            
        except Exception as e:
            logging.warning(f"Error analyzing PDF text content: {str(e)}")
            return True  # Assume scanned if analysis fails
    
    def save_pages_as_images(self, pdf_path: str, output_dir: str, 
                           page_range: Optional[tuple] = None, 
                           prefix: str = "page") -> List[str]:
        """
        Extract PDF pages and save as image files.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            page_range: Optional tuple (start, end) for page range
            prefix: Prefix for output filenames
            
        Returns:
            List of saved image file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        images = self.extract_pages_as_images(pdf_path, page_range)
        saved_paths = []
        
        for i, img in enumerate(images):
            # Determine page number for filename
            if page_range:
                page_num = page_range[0] + i + 1
            else:
                page_num = i + 1
            
            filename = f"{prefix}_{page_num:03d}.{self.output_format.lower()}"
            filepath = os.path.join(output_dir, filename)
            
            img.save(filepath, format=self.output_format, quality=95 if self.output_format == 'JPEG' else None)
            saved_paths.append(filepath)
            
        logging.info(f"Saved {len(saved_paths)} page images to {output_dir}")
        return saved_paths


def process_pdf_for_ocr(pdf_path: str, 
                       output_dir: Optional[str] = None,
                       dpi: int = 300,
                       page_range: Optional[tuple] = None) -> Dict:
    """
    Convenience function to process PDF for OCR pipeline.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Optional directory to save extracted images
        dpi: Resolution for image extraction
        page_range: Optional page range tuple
        
    Returns:
        Dictionary with PDF metadata, images, and processing info
    """
    processor = PDFProcessor(dpi=dpi)
    
    # Get PDF metadata
    metadata = processor.get_pdf_metadata(pdf_path)
    
    # Check if PDF needs OCR
    is_scanned = processor.is_scanned_pdf(pdf_path)
    
    # Extract existing text if available
    existing_text = processor.extract_text_from_pdf(pdf_path, page_range) if not is_scanned else []
    
    # Extract images for OCR
    images = processor.extract_pages_as_images(pdf_path, page_range)
    
    # Save images if output directory specified
    saved_paths = []
    if output_dir:
        saved_paths = processor.save_pages_as_images(pdf_path, output_dir, page_range)
    
    return {
        'pdf_path': pdf_path,
        'metadata': metadata,
        'is_scanned': is_scanned,
        'existing_text': existing_text,
        'images': images,
        'saved_image_paths': saved_paths,
        'total_pages': len(images),
        'processing_info': {
            'dpi': dpi,
            'page_range': page_range,
            'output_format': processor.output_format
        }
    }
