import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict, fields
import logging

from .pipeline import OcrPipelineConfig
from .batch_processor import BatchConfig


class ConfigManager:
    """Configuration management system for OCR pipeline."""
    
    SUPPORTED_FORMATS = ['.json', '.yaml', '.yml']
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_config(self, 
                   config: Union[OcrPipelineConfig, BatchConfig, Dict], 
                   name: str, 
                   format: str = 'json') -> str:
        """
        Save configuration to file.
        
        Args:
            config: Configuration object or dictionary
            name: Configuration name
            format: File format ('json' or 'yaml')
            
        Returns:
            Path to saved configuration file
        """
        if format not in ['json', 'yaml', 'yml']:
            raise ValueError(f"Unsupported format: {format}")
        
        # Convert config to dictionary if needed
        if hasattr(config, '__dict__'):
            config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
        else:
            config_dict = config
        
        # Add metadata
        config_dict['_metadata'] = {
            'config_name': name,
            'config_type': type(config).__name__ if hasattr(config, '__class__') else 'dict',
            'created_at': str(Path().cwd()),
            'format_version': '1.0'
        }
        
        # Determine file path
        filename = f"{name}.{format}"
        filepath = self.config_dir / filename
        
        # Save configuration
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if format == 'json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:  # yaml
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Configuration saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def load_config(self, 
                   name_or_path: str, 
                   config_type: Optional[type] = None) -> Union[Dict, OcrPipelineConfig, BatchConfig]:
        """
        Load configuration from file.
        
        Args:
            name_or_path: Configuration name or file path
            config_type: Expected configuration type
            
        Returns:
            Loaded configuration object
        """
        # Determine file path
        if os.path.exists(name_or_path):
            filepath = Path(name_or_path)
        else:
            # Try to find config file with supported extensions
            filepath = None
            for ext in self.SUPPORTED_FORMATS:
                candidate = self.config_dir / f"{name_or_path}{ext}"
                if candidate.exists():
                    filepath = candidate
                    break
            
            if filepath is None:
                raise FileNotFoundError(f"Configuration not found: {name_or_path}")
        
        # Load configuration
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:  # yaml
                    config_dict = yaml.safe_load(f)
            
            # Remove metadata
            metadata = config_dict.pop('_metadata', {})
            
            # Convert to specific config type if requested
            if config_type:
                if config_type == OcrPipelineConfig:
                    return self._dict_to_ocr_config(config_dict)
                elif config_type == BatchConfig:
                    return self._dict_to_batch_config(config_dict)
            
            self.logger.info(f"Configuration loaded: {filepath}")
            return config_dict
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def list_configs(self) -> Dict[str, Dict]:
        """
        List all available configurations.
        
        Returns:
            Dictionary mapping config names to metadata
        """
        configs = {}
        
        for filepath in self.config_dir.iterdir():
            if filepath.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        if filepath.suffix.lower() == '.json':
                            config_dict = json.load(f)
                        else:
                            config_dict = yaml.safe_load(f)
                    
                    metadata = config_dict.get('_metadata', {})
                    configs[filepath.stem] = {
                        'file_path': str(filepath),
                        'format': filepath.suffix[1:],
                        'size': filepath.stat().st_size,
                        'modified': filepath.stat().st_mtime,
                        **metadata
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to read config {filepath}: {e}")
        
        return configs
    
    def delete_config(self, name: str) -> bool:
        """
        Delete a configuration file.
        
        Args:
            name: Configuration name
            
        Returns:
            True if deleted successfully
        """
        for ext in self.SUPPORTED_FORMATS:
            filepath = self.config_dir / f"{name}{ext}"
            if filepath.exists():
                try:
                    filepath.unlink()
                    self.logger.info(f"Configuration deleted: {filepath}")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to delete configuration: {e}")
                    return False
        
        self.logger.warning(f"Configuration not found: {name}")
        return False
    
    def create_preset_configs(self):
        """Create preset configurations for common use cases."""
        presets = {
            'high_quality': OcrPipelineConfig(
                language_hints="eng",
                psm=3,
                oem=3,
                binarize_method="sauvola",
                denoise_method="bilateral",
                contrast_method="clahe",
                enhance_contrast=True,
                remove_shadows=True,
                detect_tables=True,
                classify_regions=True,
                include_confidence=True
            ),
            'fast_processing': OcrPipelineConfig(
                language_hints="eng",
                psm=6,
                oem=3,
                binarize_method="otsu",
                denoise_method="gaussian",
                contrast_method="histogram_eq",
                enhance_contrast=False,
                remove_shadows=False,
                detect_tables=False,
                classify_regions=False,
                include_confidence=False
            ),
            'historical_documents': OcrPipelineConfig(
                language_hints="eng",
                psm=3,
                oem=1,
                binarize_method="sauvola",
                denoise_method="non_local_means",
                contrast_method="clahe",
                enhance_contrast=True,
                remove_shadows=True,
                morphological_ops=True,
                detect_tables=True,
                classify_regions=True,
                include_confidence=True
            ),
            'multilingual': OcrPipelineConfig(
                language_hints="eng+deu+fra+spa",
                psm=3,
                oem=3,
                binarize_method="adaptive_gaussian",
                denoise_method="bilateral",
                contrast_method="clahe",
                enhance_contrast=True,
                normalize_digits=True,
                normalize_diacritics=True,
                detect_tables=True,
                include_confidence=True
            )
        }
        
        for name, config in presets.items():
            try:
                self.save_config(config, name, 'json')
                self.logger.info(f"Created preset configuration: {name}")
            except Exception as e:
                self.logger.error(f"Failed to create preset {name}: {e}")
    
    def _dict_to_ocr_config(self, config_dict: Dict) -> OcrPipelineConfig:
        """Convert dictionary to OcrPipelineConfig object."""
        # Get valid field names
        valid_fields = {f.name for f in fields(OcrPipelineConfig)}
        
        # Filter dictionary to only include valid fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return OcrPipelineConfig(**filtered_dict)
    
    def _dict_to_batch_config(self, config_dict: Dict) -> BatchConfig:
        """Convert dictionary to BatchConfig object."""
        # Get valid field names
        valid_fields = {f.name for f in fields(BatchConfig)}
        
        # Filter dictionary to only include valid fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return BatchConfig(**filtered_dict)
    
    def export_config_template(self, config_type: type, output_path: str):
        """
        Export a configuration template with all available options.
        
        Args:
            config_type: Configuration class type
            output_path: Output file path
        """
        if config_type == OcrPipelineConfig:
            template = OcrPipelineConfig()
        elif config_type == BatchConfig:
            template = BatchConfig()
        else:
            raise ValueError(f"Unsupported config type: {config_type}")
        
        # Add documentation
        template_dict = asdict(template)
        template_dict['_documentation'] = self._get_config_documentation(config_type)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.endswith('.json'):
                json.dump(template_dict, f, indent=2, ensure_ascii=False)
            else:
                yaml.dump(template_dict, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"Configuration template exported: {output_path}")
    
    def _get_config_documentation(self, config_type: type) -> Dict[str, str]:
        """Get documentation for configuration fields."""
        if config_type == OcrPipelineConfig:
            return {
                'language_hints': 'Tesseract language codes (e.g., "eng", "eng+deu")',
                'psm': 'Page segmentation mode (0-13)',
                'oem': 'OCR engine mode (0-3)',
                'binarize_method': 'Binarization method (sauvola, niblack, otsu, adaptive_gaussian, adaptive_mean)',
                'denoise_method': 'Denoising method (bilateral, non_local_means, gaussian, median)',
                'contrast_method': 'Contrast enhancement method (clahe, histogram_eq, gamma)',
                'detect_tables': 'Enable table structure detection',
                'classify_regions': 'Enable region type classification',
                'include_confidence': 'Include confidence scores in results'
            }
        elif config_type == BatchConfig:
            return {
                'max_workers': 'Maximum number of parallel workers',
                'use_multiprocessing': 'Use multiprocessing instead of threading',
                'chunk_size': 'Number of files to process in each chunk',
                'save_intermediate': 'Save intermediate results to files',
                'continue_on_error': 'Continue processing if individual files fail'
            }
        return {}


# Global config manager instance
config_manager = ConfigManager()
