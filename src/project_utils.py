import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm
import pandas as pd
import time

class ProjectManager:
    """
    Manages project-wide utility functions and directory structure.
    """
    @staticmethod
    def find_project_root():
        """
        Find the project root directory by looking for src directory.
        """
        current_dir = Path.cwd()
        while current_dir.as_posix() != '/':
            if (current_dir / 'src').exists():
                return current_dir
            current_dir = current_dir.parent
        raise ValueError("Could not find project root directory")

    @staticmethod
    def setup_project_dirs(project_root: Path) -> Dict[str, Path]:
        """
        Creates and returns project directory structure.

        Returns:
            Dict with paths for different project directories
        """
        dirs = {
            'data': project_root / 'data',
            'raw': project_root / 'data' / 'raw',
            'processed': project_root / 'data' / 'processed',
            'logs': project_root / 'logs',
            'models': project_root / 'models',
            'output': project_root / 'output'
        }

        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        return dirs


class LoggingManager:
    """
    Manages project logging configuration.
    """
    @staticmethod
    def setup_logging(project_root: Path, name: str = __name__, console_output=False) -> logging.Logger:
        """Setup logging with file and console output."""
        # Create timestamped log directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = project_root / 'logs' / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        logger = logging.getLogger(name)

        # Add file handlers
        handlers = [
            (log_dir / 'process.log', logging.INFO),
            (log_dir / 'errors.log', logging.ERROR),
        ]

        for log_file, level in handlers:
            handler = logging.FileHandler(log_file)
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                '%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(handler)

        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                '%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(console_handler)

        return logger


class ParallelProcessor:
    """
    Handles parallel processing operations.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def process_parallel(
            self,
            items: List[Any],
            process_fn: Callable,
            desc: str = "Processing",
            max_workers: Optional[int] = None,
            **kwargs
    ) -> List[Any]:
        """Process items in parallel using threads."""
        if max_workers is None:
            max_workers = os.cpu_count() * 2

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_fn, item, **kwargs)
                for item in items
            ]

            for future in tqdm(futures, desc=desc):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {str(e)}")

        return results