"""
Disney Movie ETL Pipeline
------------------------
A pipeline for extracting, transforming, and loading Disney movie data
for RAG applications.

This package provides tools for:
- Scraping movie data from Disney's website
- Processing and storing structured data
- Generating embeddings for RAG implementation
"""

from .disney_etl import DisneyETL
from .project_utils import ProjectManager, LoggingManager, ParallelProcessor
from .data_ingestion import MovieScraper
from .data_preprocessing import MoviePreprocessor
from .database_operations import DatabaseManager

__version__ = '0.1.0'

__all__ = [
    'DisneyETL',
    'MovieScraper',
    'DatabaseManager',
    'ProjectManager',
    'LoggingManager',
    'ParallelProcessor',
]