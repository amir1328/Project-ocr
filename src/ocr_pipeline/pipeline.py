import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image

from .preprocessing import preprocess_image
from .layout import detect_text_regions, analyze_document_layout
from .ocr import run_ocr_on_regions, get_ocr_statistics
from .postprocess import Postprocessor


@dataclass
class OcrPipelineConfig:
    language_hints: str = "eng"
    psm: int = 3
    oem: int = 3
    min_region_area: int = 500
    binarize: bool = True
    denoise: bool = True
    deskew: bool = True
    dewarp: bool = False
    lexicon_paths: Optional[List[str]] = None
    max_edit_distance_dictionary: int = 2
    normalize_digits: bool = True
    normalize_diacritics: bool = True
    # Enhanced preprocessing options
    enhance_contrast: bool = True
    remove_shadows: bool = False
    morphological_ops: bool = False
    contrast_method: str = "clahe"
    denoise_method: str = "bilateral"
    binarize_method: str = "sauvola"
    morph_operation: str = "opening"
    # Layout analysis options
    detect_tables: bool = True
    classify_regions: bool = True
    # OCR options
    include_confidence: bool = True


class OcrPipeline:
    def __init__(self, config: Optional[OcrPipelineConfig] = None) -> None:
        self.config = config or OcrPipelineConfig()
        self.postprocessor = Postprocessor(
            lexicon_paths=self.config.lexicon_paths or [],
            max_edit_distance_dictionary=self.config.max_edit_distance_dictionary,
            normalize_digits=self.config.normalize_digits,
            normalize_diacritics=self.config.normalize_diacritics,
        )

    def process_image(self, image_path: str) -> Dict:
        """Process an image file with enhanced OCR pipeline."""
        logging.info(f"Processing image: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        return self.process_image_direct(image, image_path)
    
    def process_image_direct(self, image: Image.Image, image_path: str = "") -> Dict:
        """Process a PIL Image directly with enhanced OCR pipeline."""
        # Enhanced preprocessing
        pre = preprocess_image(
            image,
            binarize=self.config.binarize,
            denoise=self.config.denoise,
            deskew=self.config.deskew,
            dewarp=self.config.dewarp,
            enhance_contrast=self.config.enhance_contrast,
            remove_shadows=self.config.remove_shadows,
            morphological_ops=self.config.morphological_ops,
            contrast_method=self.config.contrast_method,
            denoise_method=self.config.denoise_method,
            binarize_method=self.config.binarize_method,
            morph_operation=self.config.morph_operation,
        )
        
        # Enhanced layout analysis
        regions = detect_text_regions(
            pre, 
            min_area=self.config.min_region_area,
            detect_tables=self.config.detect_tables,
            classify_regions=self.config.classify_regions
        )

        # Fallback: if no regions, OCR the full image as one region
        if not regions:
            w, h = pre.size
            regions = [{
                "bbox": (0, 0, w, h),
                "type": "full_image",
                "region_id": 0,
                "area": w * h,
                "aspect_ratio": w / h,
                "center": (w // 2, h // 2),
                "reading_order": 0
            }]

        # Enhanced OCR with confidence scoring
        ocr_results = run_ocr_on_regions(
            pre,
            regions,
            language=self.config.language_hints,
            psm=self.config.psm,
            oem=self.config.oem,
            include_confidence=self.config.include_confidence,
        )
        
        # Extract text and create full text
        full_text = "\n".join([r["text"] for r in ocr_results if r.get("text")])
        clean_text = self.postprocessor.clean_text(full_text)
        
        # Analyze document layout
        layout_analysis = analyze_document_layout(regions)
        
        # Calculate OCR statistics
        ocr_stats = get_ocr_statistics(ocr_results) if self.config.include_confidence else {}
        
        result = {
            "image_path": image_path,
            "regions": regions,
            "ocr_results": ocr_results,
            "raw_text": full_text,
            "text": clean_text,
            "layout_analysis": layout_analysis,
            "ocr_statistics": ocr_stats,
            "processing_info": {
                "total_regions": len(regions),
                "config": {
                    "language": self.config.language_hints,
                    "psm": self.config.psm,
                    "oem": self.config.oem,
                    "preprocessing": {
                        "binarize_method": self.config.binarize_method,
                        "denoise_method": self.config.denoise_method,
                        "contrast_method": self.config.contrast_method
                    }
                }
            }
        }
        
        logging.info(f"Processing completed: {len(regions)} regions, {len(clean_text)} characters")
        return result

    def process_folder(self, input_dir: str) -> List[Dict]:
        outputs: List[Dict] = []
        for name in sorted(os.listdir(input_dir)):
            path = os.path.join(input_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                outputs.append(self.process_image(path))
            except Exception as exc:
                outputs.append({"image_path": path, "error": str(exc)})
        return outputs
