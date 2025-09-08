from typing import List, Dict
from PIL import Image
import pytesseract


def _build_tesseract_config(psm: int, oem: int) -> str:
    try:
        ver = pytesseract.get_tesseract_version()
        major = int(str(ver).split(".")[0])
    except Exception:
        major = 4  # assume modern if unknown

    if major >= 4:
        return f"--oem {oem} --psm {psm}"
    # Tesseract 3.x: no --oem, use -psm only; some builds choke on flags, so keep minimal
    return f"-psm {psm}"


def run_ocr_on_regions(
    image: Image.Image,
    regions: List[Dict],
    language: str = "eng",
    psm: int = 3,
    oem: int = 3,
) -> List[Dict]:
    results: List[Dict] = []
    config = _build_tesseract_config(psm=psm, oem=oem)
    for region in regions:
        x, y, w, h = region["bbox"]
        crop = image.crop((x, y, x + w, y + h))
        text = pytesseract.image_to_string(crop, lang=language, config=config)
        results.append({"bbox": (x, y, w, h), "text": text.strip()})
    return results
