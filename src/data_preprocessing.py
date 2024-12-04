from .project_utils import ProjectManager
from pathlib import Path
import pandas as pd
import json
from tqdm.notebook import tqdm
from typing import Dict, List


class MoviePreprocessor:
    """
    Handles data cleaning and preprocessing for Disney movie data.
    """
    def __init__(self, project_manager):
        self.project_root = project_manager.project_root
        self.logger = ProjectManager.setup_component_logging(root=self.project_root, name=self.__class__.__name__)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text fields by removing extra whitespace, special characters etc."""
        if not text:
            return ""
        text = text.strip()
        text = " ".join(text.split())  # Remove extra whitespace
        text = text.lower()
        return text

    def preprocess_movie(self, movie: Dict) -> Dict:
        """Clean and preprocess a single movie dictionary."""
        processed = {}

        # Clean text fields
        text_fields = ['title', 'description', 'rating', 'runtime']
        for field in text_fields:
            processed[field] = self.clean_text(movie.get(field, ''))

        # Handle lists (genres, cast, etc.)
        list_fields = ['genres', 'directors', 'writers', 'cast']
        for field in list_fields:
            items = movie.get(field, [])
            processed[field] = [self.clean_text(item) for item in items if item]

        # Copy non-text fields
        processed.update({
            'year': movie.get('year'),
            'slug': movie.get('slug'),
            'type': movie.get('type', '').lower(),
        })

        return processed

    def preprocess_movies(self, movies: List[Dict]) -> List[Dict]:
        """Process a list of movies from the database."""
        self.logger.info("Starting data preprocessing")
        preprocessed_movies = []

        for movie in tqdm(movies, desc="Preprocessing movies"):
            try:
                processed = self.preprocess_movie(movie)
                preprocessed_movies.append(processed)
            except Exception as e:
                self.logger.error(f"Error processing {movie.get('title', 'Unknown')}: {e}")
                continue

        self.logger.info(f"Preprocessed {len(preprocessed_movies)} movies")
        return preprocessed_movies

