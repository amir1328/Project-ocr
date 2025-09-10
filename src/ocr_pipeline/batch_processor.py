from typing import List, Dict, Optional, Callable, Union, Iterator
import logging
import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import threading
from queue import Queue

from tqdm import tqdm
from PIL import Image

from .pipeline import OcrPipeline, OcrPipelineConfig
from .pdf_processor import PDFProcessor, process_pdf_for_ocr


@dataclass
class BatchJob:
    """Represents a single batch processing job."""
    job_id: str
    input_path: str
    output_path: str
    file_type: str  # 'image' or 'pdf'
    status: str = 'pending'  # pending, processing, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    result: Optional[Dict] = None
    progress: float = 0.0


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_workers: int = 4
    use_multiprocessing: bool = False
    chunk_size: int = 10
    save_intermediate: bool = True
    continue_on_error: bool = True
    progress_callback: Optional[Callable] = None


class BatchProcessor:
    """Enhanced batch processor with progress tracking and parallel processing."""
    
    def __init__(self, 
                 ocr_config: Optional[OcrPipelineConfig] = None,
                 batch_config: Optional[BatchConfig] = None):
        """
        Initialize batch processor.
        
        Args:
            ocr_config: OCR pipeline configuration
            batch_config: Batch processing configuration
        """
        self.ocr_config = ocr_config or OcrPipelineConfig()
        self.batch_config = batch_config or BatchConfig()
        self.pipeline = OcrPipeline(self.ocr_config)
        self.pdf_processor = PDFProcessor()
        
        # Progress tracking
        self.jobs: Dict[str, BatchJob] = {}
        self.progress_queue = Queue()
        self.total_jobs = 0
        self.completed_jobs = 0
        self.failed_jobs = 0
        
        # Thread safety
        self.lock = threading.Lock()
    
    def discover_files(self, 
                      input_paths: List[str], 
                      supported_extensions: Optional[List[str]] = None) -> List[str]:
        """
        Discover all processable files from input paths.
        
        Args:
            input_paths: List of file or directory paths
            supported_extensions: List of supported file extensions
            
        Returns:
            List of discovered file paths
        """
        if supported_extensions is None:
            supported_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.pdf']
        
        discovered_files = []
        
        for input_path in input_paths:
            path = Path(input_path)
            
            if path.is_file():
                if path.suffix.lower() in supported_extensions:
                    discovered_files.append(str(path))
                else:
                    logging.warning(f"Unsupported file type: {path}")
            
            elif path.is_dir():
                for ext in supported_extensions:
                    discovered_files.extend([
                        str(p) for p in path.rglob(f"*{ext}")
                    ])
            
            else:
                logging.warning(f"Path not found: {path}")
        
        logging.info(f"Discovered {len(discovered_files)} files for processing")
        return sorted(discovered_files)
    
    def create_jobs(self, 
                   input_files: List[str], 
                   output_dir: str) -> List[BatchJob]:
        """
        Create batch jobs from input files.
        
        Args:
            input_files: List of input file paths
            output_dir: Output directory for results
            
        Returns:
            List of BatchJob objects
        """
        os.makedirs(output_dir, exist_ok=True)
        jobs = []
        
        for i, input_file in enumerate(input_files):
            input_path = Path(input_file)
            
            # Determine file type
            file_type = 'pdf' if input_path.suffix.lower() == '.pdf' else 'image'
            
            # Create output filename
            output_filename = f"{input_path.stem}_ocr_result.json"
            output_path = os.path.join(output_dir, output_filename)
            
            job = BatchJob(
                job_id=f"job_{i:04d}",
                input_path=str(input_path),
                output_path=output_path,
                file_type=file_type
            )
            
            jobs.append(job)
            self.jobs[job.job_id] = job
        
        self.total_jobs = len(jobs)
        logging.info(f"Created {len(jobs)} batch jobs")
        return jobs
    
    def process_single_job(self, job: BatchJob) -> BatchJob:
        """
        Process a single batch job.
        
        Args:
            job: BatchJob to process
            
        Returns:
            Updated BatchJob with results
        """
        job.start_time = time.time()
        job.status = 'processing'
        
        try:
            if job.file_type == 'pdf':
                result = self._process_pdf_job(job)
            else:
                result = self._process_image_job(job)
            
            job.result = result
            job.status = 'completed'
            job.progress = 1.0
            
            # Save result to file
            if self.batch_config.save_intermediate:
                with open(job.output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            
            with self.lock:
                self.completed_jobs += 1
            
            logging.info(f"Completed job {job.job_id}: {job.input_path}")
            
        except Exception as e:
            job.error_message = str(e)
            job.status = 'failed'
            job.progress = 0.0
            
            with self.lock:
                self.failed_jobs += 1
            
            logging.error(f"Failed job {job.job_id}: {e}")
            
            if not self.batch_config.continue_on_error:
                raise
        
        finally:
            job.end_time = time.time()
        
        return job
    
    def _process_image_job(self, job: BatchJob) -> Dict:
        """Process a single image file."""
        result = self.pipeline.process_image(job.input_path)
        
        # Add job metadata
        result.update({
            'job_id': job.job_id,
            'input_path': job.input_path,
            'file_type': job.file_type,
            'processing_time': time.time() - job.start_time if job.start_time else 0
        })
        
        return result
    
    def _process_pdf_job(self, job: BatchJob) -> Dict:
        """Process a single PDF file."""
        # Extract PDF data
        pdf_data = process_pdf_for_ocr(job.input_path)
        
        results = []
        images = pdf_data['images']
        
        for i, image in enumerate(images):
            # Update progress
            job.progress = i / len(images)
            
            # Process each page
            page_result = self.pipeline.process_image_direct(image)
            page_result.update({
                'page_number': i + 1,
                'total_pages': len(images)
            })
            results.append(page_result)
        
        # Combine results
        combined_result = {
            'job_id': job.job_id,
            'input_path': job.input_path,
            'file_type': job.file_type,
            'pdf_metadata': pdf_data['metadata'],
            'is_scanned': pdf_data['is_scanned'],
            'total_pages': len(images),
            'pages': results,
            'processing_time': time.time() - job.start_time if job.start_time else 0
        }
        
        return combined_result
    
    def process_batch(self, 
                     input_paths: List[str], 
                     output_dir: str,
                     show_progress: bool = True) -> Dict:
        """
        Process a batch of files with progress tracking.
        
        Args:
            input_paths: List of input file or directory paths
            output_dir: Output directory for results
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with batch processing results and statistics
        """
        start_time = time.time()
        
        # Discover files
        input_files = self.discover_files(input_paths)
        if not input_files:
            logging.warning("No files found to process")
            return {'status': 'no_files', 'results': []}
        
        # Create jobs
        jobs = self.create_jobs(input_files, output_dir)
        
        # Process jobs
        results = []
        
        if self.batch_config.use_multiprocessing:
            results = self._process_with_multiprocessing(jobs, show_progress)
        else:
            results = self._process_with_threading(jobs, show_progress)
        
        # Calculate statistics
        end_time = time.time()
        total_time = end_time - start_time
        
        statistics = {
            'total_jobs': self.total_jobs,
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'success_rate': self.completed_jobs / self.total_jobs if self.total_jobs > 0 else 0,
            'total_processing_time': total_time,
            'average_time_per_job': total_time / self.total_jobs if self.total_jobs > 0 else 0
        }
        
        # Save batch summary
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        summary = {
            'statistics': statistics,
            'job_details': [asdict(job) for job in jobs],
            'config': {
                'ocr_config': asdict(self.ocr_config),
                'batch_config': asdict(self.batch_config)
            }
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Batch processing completed: {statistics}")
        
        return {
            'status': 'completed',
            'statistics': statistics,
            'results': results,
            'summary_path': summary_path
        }
    
    def _process_with_threading(self, jobs: List[BatchJob], show_progress: bool) -> List[Dict]:
        """Process jobs using threading."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.batch_config.max_workers) as executor:
            # Submit jobs
            future_to_job = {executor.submit(self.process_single_job, job): job for job in jobs}
            
            # Process with progress bar
            if show_progress:
                with tqdm(total=len(jobs), desc="Processing files") as pbar:
                    for future in as_completed(future_to_job):
                        job = future_to_job[future]
                        try:
                            result_job = future.result()
                            if result_job.result:
                                results.append(result_job.result)
                        except Exception as e:
                            logging.error(f"Job {job.job_id} failed: {e}")
                        finally:
                            pbar.update(1)
            else:
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result_job = future.result()
                        if result_job.result:
                            results.append(result_job.result)
                    except Exception as e:
                        logging.error(f"Job {job.job_id} failed: {e}")
        
        return results
    
    def _process_with_multiprocessing(self, jobs: List[BatchJob], show_progress: bool) -> List[Dict]:
        """Process jobs using multiprocessing."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.batch_config.max_workers) as executor:
            # Submit jobs
            future_to_job = {executor.submit(self._process_job_mp, job): job for job in jobs}
            
            # Process with progress bar
            if show_progress:
                with tqdm(total=len(jobs), desc="Processing files") as pbar:
                    for future in as_completed(future_to_job):
                        job = future_to_job[future]
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                        except Exception as e:
                            logging.error(f"Job {job.job_id} failed: {e}")
                        finally:
                            pbar.update(1)
            else:
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logging.error(f"Job {job.job_id} failed: {e}")
        
        return results
    
    @staticmethod
    def _process_job_mp(job: BatchJob) -> Optional[Dict]:
        """Static method for multiprocessing job processing."""
        # Create new pipeline instance for multiprocessing
        pipeline = OcrPipeline()
        
        try:
            if job.file_type == 'pdf':
                pdf_data = process_pdf_for_ocr(job.input_path)
                results = []
                
                for i, image in enumerate(pdf_data['images']):
                    page_result = pipeline.process_image_direct(image)
                    page_result.update({
                        'page_number': i + 1,
                        'total_pages': len(pdf_data['images'])
                    })
                    results.append(page_result)
                
                return {
                    'job_id': job.job_id,
                    'input_path': job.input_path,
                    'file_type': job.file_type,
                    'pdf_metadata': pdf_data['metadata'],
                    'total_pages': len(pdf_data['images']),
                    'pages': results
                }
            else:
                result = pipeline.process_image(job.input_path)
                result.update({
                    'job_id': job.job_id,
                    'input_path': job.input_path,
                    'file_type': job.file_type
                })
                return result
                
        except Exception as e:
            logging.error(f"Multiprocessing job {job.job_id} failed: {e}")
            return None
    
    def get_progress(self) -> Dict:
        """Get current batch processing progress."""
        with self.lock:
            return {
                'total_jobs': self.total_jobs,
                'completed_jobs': self.completed_jobs,
                'failed_jobs': self.failed_jobs,
                'pending_jobs': self.total_jobs - self.completed_jobs - self.failed_jobs,
                'progress_percentage': (self.completed_jobs / self.total_jobs * 100) if self.total_jobs > 0 else 0,
                'success_rate': (self.completed_jobs / (self.completed_jobs + self.failed_jobs) * 100) if (self.completed_jobs + self.failed_jobs) > 0 else 0
            }


def process_batch_simple(input_paths: List[str], 
                        output_dir: str,
                        ocr_config: Optional[OcrPipelineConfig] = None,
                        max_workers: int = 4,
                        show_progress: bool = True) -> Dict:
    """
    Simplified batch processing function.
    
    Args:
        input_paths: List of input file or directory paths
        output_dir: Output directory for results
        ocr_config: OCR configuration
        max_workers: Number of parallel workers
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary with processing results
    """
    batch_config = BatchConfig(max_workers=max_workers)
    processor = BatchProcessor(ocr_config, batch_config)
    
    return processor.process_batch(input_paths, output_dir, show_progress)
