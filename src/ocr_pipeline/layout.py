from typing import List, Dict, Tuple, Optional
import logging
from PIL import Image
import numpy as np
import cv2
from scipy import ndimage


def _contours_from_thresh(thr, min_area: int) -> List[Dict]:
    """Extract contours from thresholded image and convert to region dictionaries."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dil = cv2.dilate(thr, kernel, iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[Dict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        regions.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "contour_area": cv2.contourArea(cnt),
            "type": "text_region"  # Default type
        })
    return regions


def _detect_table_structure(gray: np.ndarray, min_line_length: int = 50) -> Dict:
    """Detect table structure using line detection."""
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length))
    
    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine lines
    table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    
    # Find line intersections (potential table cells)
    intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
    
    return {
        'horizontal_lines': horizontal_lines,
        'vertical_lines': vertical_lines,
        'table_mask': table_mask,
        'intersections': intersections,
        'has_table_structure': np.sum(intersections) > 100  # Threshold for table detection
    }


def _detect_table_cells(table_info: Dict, min_cell_area: int = 500) -> List[Dict]:
    """Extract individual table cells from detected table structure."""
    if not table_info.get('has_table_structure', False):
        return []
    
    table_mask = table_info['table_mask']
    
    # Find contours in the table mask
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_cell_area:
            cells.append({
                'bbox': (x, y, w, h),
                'type': 'table_cell',
                'area': w * h
            })
    
    # Sort cells by position (top to bottom, left to right)
    cells.sort(key=lambda cell: (cell['bbox'][1], cell['bbox'][0]))
    return cells


def _classify_region_type(region_bbox: Tuple[int, int, int, int], 
                         image_shape: Tuple[int, int],
                         gray: np.ndarray) -> str:
    """Classify region type based on characteristics."""
    x, y, w, h = region_bbox
    aspect_ratio = w / h
    area = w * h
    image_area = image_shape[0] * image_shape[1]
    area_ratio = area / image_area
    
    # Extract region for analysis
    region = gray[y:y+h, x:x+w]
    
    # Calculate text density
    edges = cv2.Canny(region, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)
    
    # Classify based on characteristics
    if aspect_ratio > 5 and h < 50:  # Very wide and short
        return 'header'
    elif aspect_ratio < 0.2 and w < 50:  # Very tall and narrow
        return 'sidebar'
    elif area_ratio > 0.3:  # Large area
        return 'main_content'
    elif edge_density > 0.1 and aspect_ratio > 2:  # High edge density, wide
        return 'table_row'
    elif edge_density > 0.05:  # Moderate edge density
        return 'text_block'
    else:
        return 'unknown'


def detect_text_regions(image: Image.Image, 
                       min_area: int = 500,
                       detect_tables: bool = True,
                       classify_regions: bool = True) -> List[Dict]:
    """Enhanced text region detection with table detection and region classification.
    
    Args:
        image: Input PIL Image
        min_area: Minimum area for region detection
        detect_tables: Whether to detect table structures
        classify_regions: Whether to classify region types
        
    Returns:
        List of region dictionaries with enhanced metadata
    """
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    image_shape = gray.shape

    # Standard thresholding
    thr_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    regions_inv = _contours_from_thresh(thr_inv, min_area)
    regions_norm = _contours_from_thresh(thr, min_area)

    # Merge and deduplicate by bbox
    seen = set()
    regions: List[Dict] = []
    for r in regions_inv + regions_norm:
        bbox = tuple(r["bbox"])  # type: ignore
        if bbox in seen:
            continue
        seen.add(bbox)
        regions.append(r)
    
    # Table detection
    table_info = {}
    table_cells = []
    if detect_tables:
        logging.debug("Detecting table structures")
        table_info = _detect_table_structure(gray)
        if table_info.get('has_table_structure', False):
            table_cells = _detect_table_cells(table_info, min_area)
            logging.info(f"Detected {len(table_cells)} table cells")
    
    # Combine regular regions with table cells
    all_regions = regions + table_cells
    
    # Classify regions
    if classify_regions:
        logging.debug("Classifying region types")
        for region in all_regions:
            if 'type' not in region:  # Don't override table_cell type
                region_type = _classify_region_type(
                    region['bbox'], image_shape, gray
                )
                region['type'] = region_type
    
    # Add additional metadata
    for i, region in enumerate(all_regions):
        x, y, w, h = region['bbox']
        region.update({
            'region_id': i,
            'area': w * h,
            'aspect_ratio': w / h,
            'center': (x + w // 2, y + h // 2),
            'area_ratio': (w * h) / (image_shape[0] * image_shape[1])
        })
    
    # Sort regions by reading order (top to bottom, left to right)
    all_regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    
    # Add reading order index
    for i, region in enumerate(all_regions):
        region['reading_order'] = i
    
    logging.info(f"Detected {len(all_regions)} total regions ({len(regions)} text, {len(table_cells)} table cells)")
    
    return all_regions


def analyze_document_layout(regions: List[Dict]) -> Dict:
    """Analyze overall document layout and structure."""
    if not regions:
        return {}
    
    # Count region types
    type_counts = {}
    for region in regions:
        region_type = region.get('type', 'unknown')
        type_counts[region_type] = type_counts.get(region_type, 0) + 1
    
    # Calculate layout statistics
    areas = [r['area'] for r in regions]
    aspect_ratios = [r['aspect_ratio'] for r in regions]
    
    # Detect multi-column layout
    centers_x = [r['center'][0] for r in regions]
    unique_x_positions = len(set(int(x / 50) * 50 for x in centers_x))  # Group by 50px bins
    is_multi_column = unique_x_positions > 2
    
    # Detect table presence
    has_tables = any(r.get('type') == 'table_cell' for r in regions)
    table_cell_count = sum(1 for r in regions if r.get('type') == 'table_cell')
    
    return {
        'total_regions': len(regions),
        'region_types': type_counts,
        'layout_stats': {
            'avg_area': np.mean(areas),
            'total_area': sum(areas),
            'avg_aspect_ratio': np.mean(aspect_ratios),
            'area_std': np.std(areas)
        },
        'layout_features': {
            'is_multi_column': is_multi_column,
            'estimated_columns': min(unique_x_positions, 4),  # Cap at 4 columns
            'has_tables': has_tables,
            'table_cell_count': table_cell_count,
            'has_headers': type_counts.get('header', 0) > 0,
            'has_sidebars': type_counts.get('sidebar', 0) > 0
        }
    }
