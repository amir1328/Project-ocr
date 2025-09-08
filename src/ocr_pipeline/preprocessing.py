from typing import Tuple

import numpy as np
from PIL import Image
import cv2
from skimage import filters


def pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _deskew_cv(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return bgr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _denoise_cv(bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)


def _binarize_cv(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    th = filters.threshold_sauvola(gray, window_size=25)
    binary = (gray > th).astype(np.uint8) * 255
    bin_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return bin_bgr


def _dewarp_stub(bgr: np.ndarray) -> np.ndarray:
    # Placeholder: basic morphology to reduce curvature artifacts
    return bgr


def preprocess_image(
    image: Image.Image,
    binarize: bool = True,
    denoise: bool = True,
    deskew: bool = True,
    dewarp: bool = False,
) -> Image.Image:
    bgr = pil_to_cv(image)
    if deskew:
        bgr = _deskew_cv(bgr)
    if denoise:
        bgr = _denoise_cv(bgr)
    if binarize:
        bgr = _binarize_cv(bgr)
    if dewarp:
        bgr = _dewarp_stub(bgr)
    return cv_to_pil(bgr)
