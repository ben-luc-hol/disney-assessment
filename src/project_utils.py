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
    Manages project-wide utility functions, enforces project directory structure, sets up logging and .
    """
    def __init__(self):
        self.project_root = self.find_project_root()
        self.directories = self.setup_directories()

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

    @classmethod
    def setup_component_logging(cls, root, name) -> logging.Logger:
        """Setup logging """
        # Set root logger to ERROR level to suppress most output
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)

        # Create timestamped log directory for file logs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = root / 'logs' / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)

        # Add file handlers for when you need to debug
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

        return logger

    def setup_directories(self) -> Dict[str, Path]:
        """
        Creates and returns project directory structure.

        Returns:
            Dict with paths for different project directories
        """
        dirs = {
            'data': self.project_root / 'data',
            'raw': self.project_root / 'data' / 'raw',
            'processed': self.project_root / 'data' / 'processed',
            'logs': self.project_root / 'logs',
            'output': self.project_root / 'output'
        }

        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        return dirs

    def threadpool(self, items: List[Any], process_fn: Callable, desc: str = "Processing", max_workers: Optional[int] = None, **kwargs) -> List[Any]:
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
