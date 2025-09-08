from typing import List, Dict
from PIL import Image
import numpy as np
import cv2


def detect_text_regions(image: Image.Image, min_area: int = 500) -> List[Dict]:
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dil = cv2.dilate(thr, kernel, iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[Dict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        regions.append({"bbox": (int(x), int(y), int(w), int(h))})

    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return regions
