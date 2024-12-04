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

#from .disney_etl import DisneyETL
from .project_utils import ProjectManager
from .data_ingestion import MovieScraper
from .data_preprocessing import MoviePreprocessor
from .database_operations import DatabaseOps
from .embeddings import MovieEmbedder
from .search_and_retrieval import SearchAndRetrieval

__version__ = '0.2.0'

__all__ = [
    'DisneyETL',
    'MovieScraper',
    'DatabaseOps',
    'ProjectManager',
    'MovieEmbedder',
    'SearchAndRetrieval',
    'MoviePreprocessor'
]