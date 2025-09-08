import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image

from .preprocessing import preprocess_image
from .layout import detect_text_regions
from .ocr import run_ocr_on_regions
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
        image = Image.open(image_path).convert("RGB")
        pre = preprocess_image(
            image,
            binarize=self.config.binarize,
            denoise=self.config.denoise,
            deskew=self.config.deskew,
            dewarp=self.config.dewarp,
        )
        regions = detect_text_regions(pre, min_area=self.config.min_region_area)
        ocr_results = run_ocr_on_regions(
            pre,
            regions,
            language=self.config.language_hints,
            psm=self.config.psm,
            oem=self.config.oem,
        )
        full_text = "\n".join([r["text"] for r in ocr_results])
        clean_text = self.postprocessor.clean_text(full_text)
        return {
            "image_path": image_path,
            "regions": regions,
            "raw_text": full_text,
            "text": clean_text,
        }

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
