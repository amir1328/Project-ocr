from typing import List, Dict
from PIL import Image
import pytesseract
from pytesseract import TesseractError
from tempfile import NamedTemporaryFile


def _modern_config(psm: int, oem: int) -> str:
    return f"--oem {oem} --psm {psm}"


def _legacy_config(psm: int) -> str:
    return f"-psm {psm}"


def _ocr_with_temp_file(crop: Image.Image, language: str) -> str:
    # Prefer TIFF to avoid PNG libpng mismatches on very old Tesseract builds
    try:
        with NamedTemporaryFile(suffix=".tif", delete=True) as tmp:
            crop.save(tmp.name, format="TIFF")
            return pytesseract.image_to_string(tmp.name, lang=language, config="")
    except Exception:
        with NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            crop.save(tmp.name, format="PNG")
            return pytesseract.image_to_string(tmp.name, lang=language, config="")


def run_ocr_on_regions(
    image: Image.Image,
    regions: List[Dict],
    language: str = "eng",
    psm: int = 3,
    oem: int = 3,
) -> List[Dict]:
    results: List[Dict] = []
    modern_cfg = _modern_config(psm=psm, oem=oem)
    legacy_cfg = _legacy_config(psm=psm)

    for region in regions:
        x, y, w, h = region["bbox"]
        crop = image.crop((x, y, x + w, y + h))
        text = ""
        try:
            text = pytesseract.image_to_string(crop, lang=language, config=modern_cfg)
        except TesseractError:
            try:
                # Retry with legacy flags for older Tesseract (e.g., 3.x)
                text = pytesseract.image_to_string(crop, lang=language, config=legacy_cfg)
            except TesseractError:
                # Retry with file-based input and no flags for very old builds
                text = _ocr_with_temp_file(crop, language)
        results.append({"bbox": (x, y, w, h), "text": text.strip()})
    return results
