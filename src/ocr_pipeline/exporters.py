import json
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Any, Optional
import os
from pathlib import Path
import logging


class ResultExporter:
    """Export OCR results to various formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_json(self, results: Dict, output_path: str, pretty: bool = True) -> str:
        """
        Export results to JSON format.
        
        Args:
            results: OCR results dictionary
            output_path: Output file path
            pretty: Whether to format JSON with indentation
            
        Returns:
            Path to exported file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                else:
                    json.dump(results, f, ensure_ascii=False)
            
            self.logger.info(f"Results exported to JSON: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            raise
    
    def export_csv(self, results: Dict, output_path: str) -> str:
        """
        Export results to CSV format.
        
        Args:
            results: OCR results dictionary
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                headers = [
                    'region_id', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
                    'text', 'confidence', 'region_type', 'reading_order',
                    'area', 'aspect_ratio'
                ]
                writer.writerow(headers)
                
                # Write OCR results
                ocr_results = results.get('ocr_results', [])
                for result in ocr_results:
                    bbox = result.get('bbox', (0, 0, 0, 0))
                    row = [
                        result.get('region_index', ''),
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        result.get('text', '').replace('\n', ' '),
                        result.get('confidence', ''),
                        result.get('type', ''),
                        result.get('reading_order', ''),
                        result.get('area', ''),
                        result.get('aspect_ratio', '')
                    ]
                    writer.writerow(row)
            
            self.logger.info(f"Results exported to CSV: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            raise
    
    def export_xml(self, results: Dict, output_path: str) -> str:
        """
        Export results to XML format.
        
        Args:
            results: OCR results dictionary
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        try:
            # Create root element
            root = ET.Element('ocr_results')
            
            # Add metadata
            metadata = ET.SubElement(root, 'metadata')
            ET.SubElement(metadata, 'image_path').text = results.get('image_path', '')
            ET.SubElement(metadata, 'total_regions').text = str(len(results.get('regions', [])))
            
            # Add processing info
            proc_info = results.get('processing_info', {})
            processing = ET.SubElement(root, 'processing_info')
            for key, value in proc_info.items():
                if isinstance(value, dict):
                    sub_elem = ET.SubElement(processing, key)
                    for sub_key, sub_value in value.items():
                        ET.SubElement(sub_elem, sub_key).text = str(sub_value)
                else:
                    ET.SubElement(processing, key).text = str(value)
            
            # Add text content
            text_elem = ET.SubElement(root, 'text_content')
            ET.SubElement(text_elem, 'raw_text').text = results.get('raw_text', '')
            ET.SubElement(text_elem, 'clean_text').text = results.get('text', '')
            
            # Add regions
            regions_elem = ET.SubElement(root, 'regions')
            for region in results.get('regions', []):
                region_elem = ET.SubElement(regions_elem, 'region')
                region_elem.set('id', str(region.get('region_id', '')))
                region_elem.set('type', region.get('type', ''))
                
                # Bounding box
                bbox = region.get('bbox', (0, 0, 0, 0))
                bbox_elem = ET.SubElement(region_elem, 'bbox')
                ET.SubElement(bbox_elem, 'x').text = str(bbox[0])
                ET.SubElement(bbox_elem, 'y').text = str(bbox[1])
                ET.SubElement(bbox_elem, 'width').text = str(bbox[2])
                ET.SubElement(bbox_elem, 'height').text = str(bbox[3])
                
                # Properties
                ET.SubElement(region_elem, 'area').text = str(region.get('area', ''))
                ET.SubElement(region_elem, 'aspect_ratio').text = str(region.get('aspect_ratio', ''))
                ET.SubElement(region_elem, 'reading_order').text = str(region.get('reading_order', ''))
            
            # Add OCR results
            ocr_elem = ET.SubElement(root, 'ocr_results')
            for result in results.get('ocr_results', []):
                result_elem = ET.SubElement(ocr_elem, 'result')
                result_elem.set('region_id', str(result.get('region_index', '')))
                
                ET.SubElement(result_elem, 'text').text = result.get('text', '')
                ET.SubElement(result_elem, 'confidence').text = str(result.get('confidence', ''))
                
                # Quality metrics
                if 'image_quality' in result:
                    quality_elem = ET.SubElement(result_elem, 'image_quality')
                    for key, value in result['image_quality'].items():
                        ET.SubElement(quality_elem, key).text = str(value)
            
            # Add statistics
            if 'ocr_statistics' in results:
                stats_elem = ET.SubElement(root, 'statistics')
                for key, value in results['ocr_statistics'].items():
                    if isinstance(value, dict):
                        sub_elem = ET.SubElement(stats_elem, key)
                        for sub_key, sub_value in value.items():
                            ET.SubElement(sub_elem, sub_key).text = str(sub_value)
                    else:
                        ET.SubElement(stats_elem, key).text = str(value)
            
            # Pretty print XML
            rough_string = ET.tostring(root, 'unicode')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            # Remove empty lines
            pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            
            self.logger.info(f"Results exported to XML: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export XML: {e}")
            raise
    
    def export_text(self, results: Dict, output_path: str, include_metadata: bool = False) -> str:
        """
        Export results to plain text format.
        
        Args:
            results: OCR results dictionary
            output_path: Output file path
            include_metadata: Whether to include metadata in output
            
        Returns:
            Path to exported file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if include_metadata:
                    f.write(f"OCR Results for: {results.get('image_path', 'Unknown')}\n")
                    f.write(f"Total Regions: {len(results.get('regions', []))}\n")
                    f.write("=" * 50 + "\n\n")
                
                # Write clean text
                clean_text = results.get('text', '')
                f.write(clean_text)
                
                if include_metadata and 'ocr_statistics' in results:
                    stats = results['ocr_statistics']
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("OCR Statistics:\n")
                    f.write(f"Average Confidence: {stats.get('avg_confidence', 'N/A'):.1f}%\n")
                    f.write(f"Total Words: {stats.get('total_word_count', 'N/A')}\n")
                    f.write(f"Quality Score: {stats.get('avg_quality_score', 'N/A'):.2f}\n")
            
            self.logger.info(f"Results exported to text: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export text: {e}")
            raise
    
    def export_hocr(self, results: Dict, output_path: str) -> str:
        """
        Export results to hOCR format (HTML-based OCR format).
        
        Args:
            results: OCR results dictionary
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        try:
            html_content = []
            html_content.append('<!DOCTYPE html>')
            html_content.append('<html>')
            html_content.append('<head>')
            html_content.append('<title>hOCR Output</title>')
            html_content.append('<meta charset="utf-8">')
            html_content.append('<style>')
            html_content.append('.ocr_page { background-color: #f0f0f0; margin: 10px; }')
            html_content.append('.ocr_carea { border: 1px solid blue; }')
            html_content.append('.ocr_par { border: 1px solid green; }')
            html_content.append('.ocr_line { border: 1px solid red; }')
            html_content.append('.ocrx_word { border: 1px solid orange; }')
            html_content.append('</style>')
            html_content.append('</head>')
            html_content.append('<body>')
            
            # Page element
            html_content.append('<div class="ocr_page" id="page_1">')
            
            # Process each region
            for i, result in enumerate(results.get('ocr_results', [])):
                bbox = result.get('bbox', (0, 0, 0, 0))
                text = result.get('text', '').strip()
                confidence = result.get('confidence', 0)
                
                if text:
                    # Create paragraph element
                    bbox_str = f"bbox {bbox[0]} {bbox[1]} {bbox[0] + bbox[2]} {bbox[1] + bbox[3]}"
                    html_content.append(f'<p class="ocr_par" title="{bbox_str}; x_conf {confidence}">')
                    
                    # Split into lines and words
                    lines = text.split('\n')
                    for line_num, line in enumerate(lines):
                        if line.strip():
                            line_bbox = f"bbox {bbox[0]} {bbox[1]} {bbox[0] + bbox[2]} {bbox[1] + bbox[3]}"
                            html_content.append(f'<span class="ocr_line" title="{line_bbox}">')
                            
                            words = line.split()
                            for word in words:
                                word_bbox = f"bbox {bbox[0]} {bbox[1]} {bbox[0] + bbox[2]} {bbox[1] + bbox[3]}"
                                html_content.append(f'<span class="ocrx_word" title="{word_bbox}; x_conf {confidence}">{word}</span> ')
                            
                            html_content.append('</span>')
                            if line_num < len(lines) - 1:
                                html_content.append('<br>')
                    
                    html_content.append('</p>')
            
            html_content.append('</div>')
            html_content.append('</body>')
            html_content.append('</html>')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))
            
            self.logger.info(f"Results exported to hOCR: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export hOCR: {e}")
            raise
    
    def export_all_formats(self, results: Dict, output_dir: str, base_name: str) -> Dict[str, str]:
        """
        Export results to all supported formats.
        
        Args:
            results: OCR results dictionary
            output_dir: Output directory
            base_name: Base filename (without extension)
            
        Returns:
            Dictionary mapping format names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        formats = {
            'json': self.export_json,
            'csv': self.export_csv,
            'xml': self.export_xml,
            'txt': self.export_text,
            'hocr': self.export_hocr
        }
        
        for format_name, export_func in formats.items():
            try:
                output_path = os.path.join(output_dir, f"{base_name}.{format_name}")
                if format_name == 'txt':
                    exported_files[format_name] = export_func(results, output_path, include_metadata=True)
                else:
                    exported_files[format_name] = export_func(results, output_path)
            except Exception as e:
                self.logger.error(f"Failed to export {format_name}: {e}")
        
        return exported_files


def export_batch_results(batch_results: List[Dict], output_dir: str, formats: List[str] = None) -> Dict:
    """
    Export batch processing results to multiple formats.
    
    Args:
        batch_results: List of OCR result dictionaries
        output_dir: Output directory
        formats: List of formats to export (default: all)
        
    Returns:
        Dictionary with export statistics
    """
    if formats is None:
        formats = ['json', 'csv', 'xml', 'txt']
    
    exporter = ResultExporter()
    os.makedirs(output_dir, exist_ok=True)
    
    exported_count = 0
    failed_count = 0
    
    for i, result in enumerate(batch_results):
        try:
            base_name = f"result_{i:04d}"
            if 'image_path' in result:
                base_name = Path(result['image_path']).stem
            
            for format_name in formats:
                output_path = os.path.join(output_dir, f"{base_name}.{format_name}")
                
                if format_name == 'json':
                    exporter.export_json(result, output_path)
                elif format_name == 'csv':
                    exporter.export_csv(result, output_path)
                elif format_name == 'xml':
                    exporter.export_xml(result, output_path)
                elif format_name == 'txt':
                    exporter.export_text(result, output_path, include_metadata=True)
                elif format_name == 'hocr':
                    exporter.export_hocr(result, output_path)
            
            exported_count += 1
            
        except Exception as e:
            logging.error(f"Failed to export result {i}: {e}")
            failed_count += 1
    
    return {
        'total_results': len(batch_results),
        'exported_successfully': exported_count,
        'export_failures': failed_count,
        'output_directory': output_dir,
        'formats': formats
    }
