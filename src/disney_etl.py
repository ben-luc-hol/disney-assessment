from pathlib import Path
from typing import Dict, Any, Optional
import logging
import pandas as pd
from datetime import datetime
from .project_utils import ProjectManager, LoggingManager, ParallelProcessor
from .data_ingestion import MovieScraper
from .data_preprocessing import MoviePreprocessor
from .database_operations import DatabaseManager
from .embeddings import MovieEmbedder
from .search_and_retrieval import SearchAndRetrieval


class DisneyETL:
    """
    End-to-end orchestration class for ETL pipeline with example data processing and RAG system.

    This pipeline:
    - Extracts movie data from Disney's online movie listings
    - Transforms and preprocesses data into structured format
    - Loads data into SQL and vector databases
    - Creates embeddings for RAG implementation
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.embedder = None
        self.searcher = None
        self.project_root = ProjectManager.find_project_root()
        self.dirs = ProjectManager.setup_project_dirs(self.project_root)
        self.logger = LoggingManager.setup_logging(self.project_root, name='DisneyETL', console_output=False)

        self.config = {
            'parallel_workers': 4,
            'batch_size': 100,
            'use_processes': False,
            'db_path': self.dirs['processed'] / 'disney_movies.db',
            'embedding_model': 'text-embedding-ada-002',  # OpenAI's embedding model
            'chunk_size': 500,  # For text splitting
            'chunk_overlap': 50,
            'vector_store_path': self.dirs['processed'] / 'vector_store'
        }
        if config:
            self.config.update(config)

        self.parallel_processor = ParallelProcessor(self.logger)
        self.db_manager = DatabaseManager(db_path=self.config['db_path'], logger=self.logger)
        self.scraper = MovieScraper(logger=self.logger, parallel_processor=self.parallel_processor, movie_data_dir=self.dirs['raw'])

        # Pipeline state tracking
        self.state = {
            'extraction_complete': False,
            'transformation_complete': False,
            'sql_load_complete': False,
            'embedding_complete': False,
            'vector_store_complete': False,
            'last_run': None,
            'errors': []
        }

        # Pipeline state tracking
        self.state = {
            'collection_complete': False,
            'transformation_complete': False,
            'sql_load_complete': False,
            'embedding_complete': False,
            'vector_store_complete': False,
            'last_run': None,
            'movies_processed': 0,
            'errors': []
        }

    def collect_data(self) -> bool:
        """
        Collect movie data using the MovieScraper.

        Returns:
        bool: True if successful, False otherwise
        """
        try:
            self.logger.info("Starting data collection process")
            self.scraper.run()
            self.state['collection_complete'] = True
            self.logger.info("Data collection completed.")
            return True

        except Exception as e:
            self.logger.error(f"Error in data collection: {str(e)}")
            self.state['errors'].append(str(e))
            return False

    def preprocess_data(self) -> bool:
        """
        Preprocess scraped data before loading into database.
        """
        try:
            self.logger.info("Starting data preprocessing")

            # Initialize preprocessor
            preprocessor = MoviePreprocessor(
                logger=self.logger,
                raw_data_dir=self.dirs['raw']
            )

            # Run preprocessing - this will:
            # 1. Clean and process all JSON files
            # 2. Save processed JSONs to movie-data-preprocessed/
            # 3. Create analysis CSV
            # 4. Log statistics about the dataset
            df = preprocessor.preprocess_data()

            # Update pipeline state
            self.state['preprocessing_complete'] = True
            return True

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            self.state['errors'].append(str(e))
            return False

    def load_to_sqlite(self) -> bool:
        """
        Loads data to SQLite database.
        """
        if not self.state['preprocessing_complete']:
            self.preprocess_data()

        try:
            self.logger.info("Starting data loading process")

            self.db_manager.initialize_database()

            movie_data_dir = self.dirs['raw'] / 'movie-data-preprocessed'

            processed, failed = self.db_manager.batch_insert_from_json_files(movie_data_dir)


            self.logger.info(f"Loaded {processed} movies into database, {failed} failed")

            self.state['sql_load_complete'] = True

            return True

        except Exception as e:
            self.logger.error(f"Database loading failed: {e}")
            self.state['errors'].append(str(e))
            return False

    def vectorize_data(self) -> bool:
        """
        Create T5 embeddings for all preprocessed movies.
        """
        try:
            self.logger.info("Starting embedding creation")

            self.embedder = MovieEmbedder(
                logger=self.logger,
                preprocessed_dir=self.dirs['raw'] / 'movie-data-preprocessed',
                embeddings_dir=self.dirs['processed'] / 'embeddings',
                model_name='t5-base'
            )

            self.embedder.process_embeddings()

            self.state['embedding_complete'] = True
            return True

        except Exception as e:
            self.logger.error(f"Embedding creation failed: {e}")
            self.state['errors'].append(str(e))
            return False

    def setup_search(self) -> bool:
        """
        Initialize search components.
        """
        try:
            self.logger.info("Setting up search functionality")
            output_dir = self.dirs['output']

            # Create embedder if not exists
            if not hasattr(self, 'embedder'):
                self.embedder = MovieEmbedder(
                    logger=self.logger,
                    preprocessed_dir=self.dirs['raw'] / 'movie-data-preprocessed',
                    embeddings_dir=self.dirs['processed'] / 'embeddings',
                    model_name='t5-base'
                )

            # Initialize search and retrieval
            self.searcher = SearchAndRetrieval(
                output_dir,
                logger=self.logger,
                embedder=self.embedder,
                db_manager=self.db_manager
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup search: {e}")
            self.state['errors'].append(str(e))
            return False

    def movie_search(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Search for movies using semantic search.
        """
        try:
            if not hasattr(self, 'searcher'):
                self.setup_search()

            results = self.searcher.semantic_search(query, k)

            # Convert to DataFrame
            df = pd.DataFrame(results)
            cols = ['title', 'year', 'similarity_score', 'description', 'genres',
                    'directors', 'cast', 'rating', 'runtime']
            df = df[[c for c in cols if c in df.columns]]

            return df

        except Exception as e:
            self.logger.error(f"Movie search failed: {e}")
            raise

    def ask_about_movies(self, query: str, k: int = 5) -> str:
        """
        Ask questions or get recommendations about Disney movies.
        """
        try:
            if not hasattr(self, 'searcher'):
                self.setup_search()

            return self.searcher.generate_movie_response(query, k)

        except Exception as e:
            self.logger.error(f"Movie query failed: {e}")
            raise