import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging configuration for the OCR pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file name (optional)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        max_file_size: Maximum size for log files before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Create log directory
    if enable_file:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger("ocr_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        from logging.handlers import RotatingFileHandler
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"ocr_pipeline_{timestamp}.log"
        
        log_path = os.path.join(log_dir, log_file)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f"ocr_pipeline.{name}")


class OCRError(Exception):
    """Base exception class for OCR pipeline errors."""
    pass


class PreprocessingError(OCRError):
    """Exception raised during image preprocessing."""
    pass


class LayoutAnalysisError(OCRError):
    """Exception raised during layout analysis."""
    pass


class OCRProcessingError(OCRError):
    """Exception raised during OCR processing."""
    pass


class PostprocessingError(OCRError):
    """Exception raised during text postprocessing."""
    pass


class PDFProcessingError(OCRError):
    """Exception raised during PDF processing."""
    pass


class BatchProcessingError(OCRError):
    """Exception raised during batch processing."""
    pass


def log_performance(func):
    """Decorator to log function performance metrics."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("performance")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    
    return wrapper


def log_memory_usage(func):
    """Decorator to log memory usage of functions."""
    import functools
    import psutil
    import os
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("memory")
        process = psutil.Process(os.getpid())
        
        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_diff = mem_after - mem_before
            
            logger.debug(f"{func.__name__} memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB (Δ{mem_diff:+.1f}MB)")
            return result
        except Exception as e:
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_diff = mem_after - mem_before
            logger.warning(f"{func.__name__} failed, memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB (Δ{mem_diff:+.1f}MB)")
            raise
    
    return wrapper


def create_error_report(error: Exception, context: dict = None) -> dict:
    """Create a detailed error report for debugging."""
    import traceback
    import sys
    
    error_report = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'context': context or {}
    }
    
    return error_report


def log_system_info():
    """Log system information for debugging purposes."""
    import platform
    import psutil
    
    logger = get_logger("system")
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    memory = psutil.virtual_memory()
    logger.info(f"Memory: {memory.total / 1024**3:.1f}GB total, {memory.available / 1024**3:.1f}GB available")
    
    logger.info("=== End System Information ===")


# Initialize default logger
default_logger = setup_logging()
