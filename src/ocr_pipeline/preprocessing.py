from typing import Tuple, Optional
import logging

import numpy as np
from PIL import Image, ImageEnhance
import cv2
from skimage import filters, exposure, restoration, morphology
from scipy import ndimage


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


def _enhance_contrast(bgr: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Enhance image contrast using various methods."""
    if method == "clahe":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif method == "histogram_eq":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    elif method == "gamma":
        # Gamma correction for better contrast
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(bgr, table)
    return bgr


def _advanced_denoise(bgr: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Advanced denoising with multiple methods."""
    if method == "bilateral":
        return cv2.bilateralFilter(bgr, 9, 75, 75)
    elif method == "non_local_means":
        return cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
    elif method == "gaussian":
        return cv2.GaussianBlur(bgr, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(bgr, 5)
    return bgr


def _remove_shadows(bgr: np.ndarray) -> np.ndarray:
    """Remove shadows and uneven illumination."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)


def _morphological_operations(bgr: np.ndarray, operation: str = "opening") -> np.ndarray:
    """Apply morphological operations to clean up the image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    if operation == "opening":
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation == "closing":
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    elif operation == "gradient":
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    else:
        result = gray
    
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def _dewarp_advanced(bgr: np.ndarray) -> np.ndarray:
    """Advanced dewarping using contour detection and perspective correction."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return bgr
    
    # Find the largest contour (assuming it's the document)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we have 4 points, apply perspective correction
    if len(approx) == 4:
        # Order the points: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        # Compute the width and height of the new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Construct the destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(bgr, M, (maxWidth, maxHeight))
        return warped
    
    return bgr


def _adaptive_binarization(bgr: np.ndarray, method: str = "sauvola") -> np.ndarray:
    """Advanced binarization methods."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    if method == "sauvola":
        th = filters.threshold_sauvola(gray, window_size=25, k=0.2)
        binary = (gray > th).astype(np.uint8) * 255
    elif method == "niblack":
        th = filters.threshold_niblack(gray, window_size=25, k=0.2)
        binary = (gray > th).astype(np.uint8) * 255
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive_gaussian":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == "adaptive_mean":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        # Default to Sauvola
        th = filters.threshold_sauvola(gray, window_size=25)
        binary = (gray > th).astype(np.uint8) * 255
    
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def preprocess_image(
    image: Image.Image,
    binarize: bool = True,
    denoise: bool = True,
    deskew: bool = True,
    dewarp: bool = False,
    enhance_contrast: bool = True,
    remove_shadows: bool = False,
    morphological_ops: bool = False,
    contrast_method: str = "clahe",
    denoise_method: str = "bilateral",
    binarize_method: str = "sauvola",
    morph_operation: str = "opening",
) -> Image.Image:
    """Enhanced preprocessing pipeline with advanced image processing techniques.
    
    Args:
        image: Input PIL Image
        binarize: Apply binarization
        denoise: Apply denoising
        deskew: Apply deskewing
        dewarp: Apply dewarping/perspective correction
        enhance_contrast: Apply contrast enhancement
        remove_shadows: Remove shadows and uneven illumination
        morphological_ops: Apply morphological operations
        contrast_method: Method for contrast enhancement ('clahe', 'histogram_eq', 'gamma')
        denoise_method: Method for denoising ('bilateral', 'non_local_means', 'gaussian', 'median')
        binarize_method: Method for binarization ('sauvola', 'niblack', 'otsu', 'adaptive_gaussian', 'adaptive_mean')
        morph_operation: Morphological operation ('opening', 'closing', 'gradient')
    
    Returns:
        Preprocessed PIL Image
    """
    logging.info(f"Starting preprocessing with methods: contrast={contrast_method}, denoise={denoise_method}, binarize={binarize_method}")
    
    bgr = pil_to_cv(image)
    original_shape = bgr.shape
    
    try:
        # Step 1: Remove shadows and uneven illumination
        if remove_shadows:
            logging.debug("Applying shadow removal")
            bgr = _remove_shadows(bgr)
        
        # Step 2: Enhance contrast
        if enhance_contrast:
            logging.debug(f"Applying contrast enhancement: {contrast_method}")
            bgr = _enhance_contrast(bgr, method=contrast_method)
        
        # Step 3: Deskew the image
        if deskew:
            logging.debug("Applying deskewing")
            bgr = _deskew_cv(bgr)
        
        # Step 4: Advanced denoising
        if denoise:
            logging.debug(f"Applying denoising: {denoise_method}")
            bgr = _advanced_denoise(bgr, method=denoise_method)
        
        # Step 5: Morphological operations
        if morphological_ops:
            logging.debug(f"Applying morphological operations: {morph_operation}")
            bgr = _morphological_operations(bgr, operation=morph_operation)
        
        # Step 6: Advanced binarization
        if binarize:
            logging.debug(f"Applying binarization: {binarize_method}")
            bgr = _adaptive_binarization(bgr, method=binarize_method)
        
        # Step 7: Advanced dewarping
        if dewarp:
            logging.debug("Applying advanced dewarping")
            bgr = _dewarp_advanced(bgr)
        
        logging.info(f"Preprocessing completed. Shape: {original_shape} -> {bgr.shape}")
        return cv_to_pil(bgr)
        
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        # Return original image if preprocessing fails
        return image
