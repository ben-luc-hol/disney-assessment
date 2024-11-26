from .project_utils import logging
from typing import List, Dict
from pathlib import Path
import pandas as pd
import json


class MoviePreprocessor:
    """
    Handles data cleaning and preprocessing for Disney movie data.
    """
    def __init__(self, logger: logging.Logger, raw_data_dir):
        self.raw_dir = raw_data_dir
        self.unprocessed_dir = self.raw_dir / 'movie-data'
        self.preprocessed_dir = self.raw_dir / 'movie-data-preprocessed'
        self.preprocessed_dir.mkdir(exist_ok=True)
        self.logger = logger

    def clean_text(self, text: str) -> str:
        """
        Clean text fields by removing extra whitespace, special characters etc.
        """
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
            'release_date': movie.get('release_date'),
            'slug': movie.get('slug'),
            'type': movie.get('type', '').lower(),
            'link': movie.get('link'),
            'animation': movie.get('animation', 0)
        })

        return processed

    def output_csv(self, list_of_dicts: List[Dict]) -> pd.DataFrame:

        df = pd.DataFrame(list_of_dicts)

        # Handle list columns (flatten for CSV)
        for col in ['genres', 'directors', 'writers', 'cast']:
            df[col] = df[col].apply(lambda x: '|'.join(x) if x else '')

        # Save CSV
        output_file = self.raw_dir / 'disney_movies_analysis.csv'
        df.to_csv(output_file, index=False)
        self.logger.info(f"Created analysis CSV at {output_file}")
        return df

    def preprocess_data(self):
        """
        Process all movie data, clean it, and save both preprocessed JSON files and a CSV.
        Returns DataFrame of processed data.
        """
        self.logger.info("Starting data preprocessing")
        all_movies = []

        # Process each JSON file
        for json_file in self.unprocessed_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    movie = json.load(f)

                # Preprocess the movie data
                processed = self.preprocess_movie(movie)
                all_movies.append(processed)

                # Save preprocessed JSON
                output_file = self.preprocessed_dir / json_file.name
                with open(output_file, 'w') as f:
                    json.dump(processed, f, indent=4)

            except Exception as e:
                self.logger.error(f"Error processing {json_file.name}: {e}")
                continue

        df = self.output_csv(all_movies)
        self.logger.info(f"\nPreprocessing Summary:")
        self.logger.info(f"Total movies processed: {len(df)}")
        self.logger.info(f"Years covered: {df['year'].min()} to {df['year'].max()}")
        self.logger.info(f"Animation movies: {df['animation'].sum()}")
        self.logger.info(f"Average description length: {df['description'].str.len().mean():.0f} characters")
        self.logger.info(f"\nFiles saved to: {self.preprocessed_dir}")
        return df



