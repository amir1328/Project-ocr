from typing import List, Dict, Tuple
import logging
import re
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import TesseractError, Output
from tempfile import NamedTemporaryFile
import cv2


def _modern_config(psm: int, oem: int) -> str:
    return f"--oem {oem} --psm {psm}"


def _legacy_config(psm: int) -> str:
    return f"-psm {psm}"


def _calculate_image_quality_metrics(image: Image.Image) -> Dict[str, float]:
    """Calculate image quality metrics for OCR assessment."""
    # Convert to grayscale for analysis
    gray = np.array(image.convert('L'))
    
    # Calculate sharpness using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate contrast using standard deviation
    contrast = gray.std()
    
    # Calculate brightness (mean intensity)
    brightness = gray.mean()
    
    # Calculate noise level using high-frequency content
    noise_level = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)))
    
    # Calculate text-to-background ratio
    hist = np.histogram(gray, bins=256, range=(0, 256))[0]
    # Find peaks (assuming text and background create bimodal distribution)
    peaks = np.where(hist > np.mean(hist) + np.std(hist))[0]
    text_bg_ratio = len(peaks) / 256.0
    
    return {
        'sharpness': float(laplacian_var),
        'contrast': float(contrast),
        'brightness': float(brightness),
        'noise_level': float(noise_level),
        'text_bg_ratio': float(text_bg_ratio)
    }


def _calculate_text_quality_metrics(text: str) -> Dict[str, float]:
    """Calculate text quality metrics."""
    if not text.strip():
        return {
            'word_count': 0,
            'char_count': 0,
            'avg_word_length': 0.0,
            'special_char_ratio': 0.0,
            'digit_ratio': 0.0,
            'uppercase_ratio': 0.0,
            'whitespace_ratio': 0.0
        }
    
    words = text.split()
    chars = list(text)
    
    # Basic counts
    word_count = len(words)
    char_count = len(text)
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    
    # Character type ratios
    special_chars = sum(1 for c in chars if not c.isalnum() and not c.isspace())
    digits = sum(1 for c in chars if c.isdigit())
    uppercase = sum(1 for c in chars if c.isupper())
    whitespace = sum(1 for c in chars if c.isspace())
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'special_char_ratio': special_chars / max(1, char_count),
        'digit_ratio': digits / max(1, char_count),
        'uppercase_ratio': uppercase / max(1, char_count),
        'whitespace_ratio': whitespace / max(1, char_count)
    }


def _ocr_with_confidence(crop: Image.Image, language: str, config: str) -> Tuple[str, float, Dict]:
    """Perform OCR with confidence scoring and detailed data."""
    try:
        # Get detailed OCR data with confidence scores
        data = pytesseract.image_to_data(crop, lang=language, config=config, output_type=Output.DICT)
        
        # Extract text and confidence scores
        texts = []
        confidences = []
        
        for i, conf in enumerate(data['conf']):
            if int(conf) > 0:  # Only include confident detections
                text = data['text'][i].strip()
                if text:  # Only include non-empty text
                    texts.append(text)
                    confidences.append(int(conf))
        
        full_text = ' '.join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calculate additional metrics
        image_quality = _calculate_image_quality_metrics(crop)
        text_quality = _calculate_text_quality_metrics(full_text)
        
        detailed_data = {
            'word_confidences': confidences,
            'word_texts': texts,
            'image_quality': image_quality,
            'text_quality': text_quality,
            'tesseract_data': data
        }
        
        return full_text, avg_confidence, detailed_data
        
    except Exception as e:
        logging.warning(f"Confidence OCR failed: {e}, falling back to basic OCR")
        # Fallback to basic OCR
        try:
            text = pytesseract.image_to_string(crop, lang=language, config=config)
            image_quality = _calculate_image_quality_metrics(crop)
            text_quality = _calculate_text_quality_metrics(text)
            
            detailed_data = {
                'word_confidences': [],
                'word_texts': [],
                'image_quality': image_quality,
                'text_quality': text_quality,
                'tesseract_data': {}
            }
            
            return text, 0.0, detailed_data
        except Exception:
            return "", 0.0, {}


def _ocr_with_temp_file(crop: Image.Image, language: str) -> Tuple[str, float, Dict]:
    """OCR with temporary file fallback, including confidence scoring."""
    # Prefer TIFF to avoid PNG libpng mismatches on very old Tesseract builds
    try:
        with NamedTemporaryFile(suffix=".tif", delete=True) as tmp:
            crop.save(tmp.name, format="TIFF")
            text = pytesseract.image_to_string(tmp.name, lang=language, config="")
            image_quality = _calculate_image_quality_metrics(crop)
            text_quality = _calculate_text_quality_metrics(text)
            
            detailed_data = {
                'word_confidences': [],
                'word_texts': [],
                'image_quality': image_quality,
                'text_quality': text_quality,
                'tesseract_data': {}
            }
            
            return text, 0.0, detailed_data
    except Exception:
        with NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            crop.save(tmp.name, format="PNG")
            text = pytesseract.image_to_string(tmp.name, lang=language, config="")
            image_quality = _calculate_image_quality_metrics(crop)
            text_quality = _calculate_text_quality_metrics(text)
            
            detailed_data = {
                'word_confidences': [],
                'word_texts': [],
                'image_quality': image_quality,
                'text_quality': text_quality,
                'tesseract_data': {}
            }
            
            return text, 0.0, detailed_data


def run_ocr_on_regions(
    image: Image.Image,
    regions: List[Dict],
    language: str = "eng",
    psm: int = 3,
    oem: int = 3,
    include_confidence: bool = True,
) -> List[Dict]:
    """Enhanced OCR with confidence scoring and quality metrics.
    
    Args:
        image: Input PIL Image
        regions: List of region dictionaries with 'bbox' key
        language: Tesseract language code
        psm: Page segmentation mode
        oem: OCR engine mode
        include_confidence: Whether to include confidence scores and quality metrics
    
    Returns:
        List of dictionaries with OCR results, confidence scores, and quality metrics
    """
    results: List[Dict] = []
    modern_cfg = _modern_config(psm=psm, oem=oem)
    legacy_cfg = _legacy_config(psm=psm)
    
    logging.info(f"Processing {len(regions)} regions with language={language}, psm={psm}, oem={oem}")

    for i, region in enumerate(regions):
        x, y, w, h = region["bbox"]
        crop = image.crop((x, y, x + w, y + h))
        
        text = ""
        confidence = 0.0
        detailed_data = {}
        
        try:
            if include_confidence:
                text, confidence, detailed_data = _ocr_with_confidence(crop, language, modern_cfg)
            else:
                text = pytesseract.image_to_string(crop, lang=language, config=modern_cfg)
                
        except TesseractError as e:
            logging.warning(f"Modern OCR failed for region {i}: {e}")
            try:
                # Retry with legacy flags for older Tesseract (e.g., 3.x)
                if include_confidence:
                    text, confidence, detailed_data = _ocr_with_confidence(crop, language, legacy_cfg)
                else:
                    text = pytesseract.image_to_string(crop, lang=language, config=legacy_cfg)
                    
            except TesseractError as e2:
                logging.warning(f"Legacy OCR failed for region {i}: {e2}")
                # Retry with file-based input and no flags for very old builds
                text, confidence, detailed_data = _ocr_with_temp_file(crop, language)
        
        # Build result dictionary
        result = {
            "bbox": (x, y, w, h),
            "text": text.strip(),
            "region_index": i
        }
        
        if include_confidence:
            result.update({
                "confidence": confidence,
                "image_quality": detailed_data.get('image_quality', {}),
                "text_quality": detailed_data.get('text_quality', {}),
                "word_confidences": detailed_data.get('word_confidences', []),
                "word_texts": detailed_data.get('word_texts', [])
            })
            
            # Calculate overall quality score
            quality_score = _calculate_overall_quality_score(
                confidence, 
                detailed_data.get('image_quality', {}),
                detailed_data.get('text_quality', {})
            )
            result["quality_score"] = quality_score
        
        results.append(result)
        logging.debug(f"Region {i}: confidence={confidence:.1f}, text_length={len(text)}")
    
    logging.info(f"OCR completed for {len(results)} regions")
    return results


def _calculate_overall_quality_score(confidence: float, image_quality: Dict, text_quality: Dict) -> float:
    """Calculate an overall quality score combining multiple metrics."""
    if not image_quality or not text_quality:
        return confidence / 100.0  # Normalize confidence to 0-1 range
    
    # Normalize confidence (0-100 to 0-1)
    conf_score = confidence / 100.0
    
    # Image quality components (normalize and weight)
    sharpness_score = min(1.0, image_quality.get('sharpness', 0) / 1000.0)  # Normalize sharpness
    contrast_score = min(1.0, image_quality.get('contrast', 0) / 100.0)     # Normalize contrast
    
    # Text quality components
    word_count = text_quality.get('word_count', 0)
    word_score = min(1.0, word_count / 50.0)  # Normalize word count (50 words = perfect)
    
    # Penalize excessive special characters (might indicate OCR errors)
    special_char_penalty = max(0, text_quality.get('special_char_ratio', 0) - 0.1) * 2
    
    # Combine scores with weights
    overall_score = (
        conf_score * 0.4 +           # 40% confidence
        sharpness_score * 0.2 +      # 20% sharpness
        contrast_score * 0.2 +       # 20% contrast
        word_score * 0.2             # 20% text content
    ) - special_char_penalty
    
    return max(0.0, min(1.0, overall_score))  # Clamp to 0-1 range


def get_ocr_statistics(results: List[Dict]) -> Dict:
    """Calculate comprehensive statistics from OCR results."""
    if not results:
        return {}
    
    confidences = [r.get('confidence', 0) for r in results if 'confidence' in r]
    quality_scores = [r.get('quality_score', 0) for r in results if 'quality_score' in r]
    text_lengths = [len(r.get('text', '')) for r in results]
    word_counts = [r.get('text_quality', {}).get('word_count', 0) for r in results]
    
    stats = {
        'total_regions': len(results),
        'total_text_length': sum(text_lengths),
        'total_word_count': sum(word_counts),
        'avg_text_length_per_region': np.mean(text_lengths) if text_lengths else 0,
        'avg_words_per_region': np.mean(word_counts) if word_counts else 0
    }
    
    if confidences:
        stats.update({
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'confidence_std': np.std(confidences)
        })
    
    if quality_scores:
        stats.update({
            'avg_quality_score': np.mean(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'quality_score_std': np.std(quality_scores)
        })
    
    # Quality assessment
    high_quality_regions = sum(1 for score in quality_scores if score > 0.7)
    medium_quality_regions = sum(1 for score in quality_scores if 0.4 <= score <= 0.7)
    low_quality_regions = sum(1 for score in quality_scores if score < 0.4)
    
    stats.update({
        'high_quality_regions': high_quality_regions,
        'medium_quality_regions': medium_quality_regions,
        'low_quality_regions': low_quality_regions,
        'quality_distribution': {
            'high': high_quality_regions / len(results) if results else 0,
            'medium': medium_quality_regions / len(results) if results else 0,
            'low': low_quality_regions / len(results) if results else 0
        }
    })
    
    return stats
