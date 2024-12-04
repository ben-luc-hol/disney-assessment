import requests
import json
import time
import os
from typing import Dict, List, Optional
import json
import requests
from typing import Dict, List
from pathlib import Path
from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
import time
import logging
from .project_utils import ProjectManager


class MovieScraper:
    """
    Class to ingest movie title data directly from Disney's website.
    """
    base_url = "https://movies.disney.com/_grill/filter/all-movies"

    def __init__(self, project_manager):
        self.project_root = project_manager.project_root
        self.logger = ProjectManager.setup_component_logging(root=self.project_root, name=self.__class__.__name__)
        self.movie_dir = project_manager.directories['raw'] / "movie-data"
        self.movie_dir.mkdir(exist_ok=True)

    def get_all_movies(self) -> List[Dict]:
        """
        Paginate through the API to get all movies.
        Returns list of movie dictionaries from API.
        """
        self.logger.info("Calling API to get all movies")
        all_movies = []
        offset = 0

        # Get first page to get total count
        response = requests.get(
            self.base_url,
            params={"filter": "A-Z", "mod": 0, "offset": 0, "slug": "a-z"}
        )
        data = response.json()
        total_count = data['count']
        self.logger.info(f"Found {total_count} total movies to process")

        # Calculate number of pages needed
        pages_needed = (total_count + 39) // 40  # Round up division

        # Paginate through all pages
        for page in tqdm(range(pages_needed), desc="Iterating through Disney's API by page"):
            offset = page * 40
            response = requests.get(
                self.base_url,
                params={"filter": "A-Z", "mod": 0, "offset": offset, "slug": "a-z"}
            )
            data = response.json()

            if not data.get('data'):
                break

            all_movies.extend(data['data'])
            self.logger.debug(f"Processed page {page + 1}/{pages_needed}")
            time.sleep(0.3)  # Be nice to Disney's servers

        self.logger.info(f"Successfully collected {len(all_movies)} movie titles")
        return all_movies

    def movie_to_json(self, movie_dict: Dict):
        """
        Saves a single movie dictionary to a JSON file by using the movie's slugified title
        """
        if not movie_dict:
            self.logger.warning("Received None or empty movie dictionary")
            return

        # Generate filename, with fallback if slug is missing
        if 'slug' not in movie_dict:
            # Use title if available, otherwise use timestamp
            if 'title' in movie_dict:
                safe_title = "".join(c for c in movie_dict['title'] if c.isalnum() or c in (' ', '-')).lower().replace(
                    ' ', '-')
                filename = f"{safe_title}.json"
            else:
                filename = f"movie_{int(time.time())}.json"
            self.logger.warning(f"Movie missing slug, using filename: {filename}")
        else:
            filename = f"{movie_dict['slug']}.json"

        # Save the file
        try:
            filepath = self.movie_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(movie_dict, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving movie to JSON: {str(e)}")

    def process_movie(self, movie: Dict):
        """
        Processes a single movie dictionary element from the API:
        1. Pulls relevant keys from full metadata dictionary
        2. Visits the movie's href page
        2. Extracts and verifies metadata (ratings, genres, cast etc.)
        3. Return enhanced movie dictionary
        """
        movie_dict = {
            "title": movie.get('title'),
            "year": movie.get('year'),
            "release_date": movie['raw_dates'].get('release'),
            "description": movie.get('description'),
            "slug": movie.get('slug'),
            "type": movie.get('type').lower(),
            "link": movie.get('href')
        }

        movie_dict = self.scrape_additional_data(movie_dict)

        self.movie_to_json(movie_dict)

    def scrape_additional_data(self, movie_dict: Dict) -> Dict:
        """
        Scrapes additional data about a movie
        :param movie_dict: Raw dictionary of movie data pulled as JSON structure from Disney's API
        :return updated_movie_dict: Clean movie dictionary
        """
        try:
            response = requests.get(movie_dict['link'])

            soup = BeautifulSoup(response.text, 'html.parser')

            # Select movie data section
            updated_movie_dict = movie_dict

            metadata = soup.select_one('.meta-compact')
            if metadata:
                # Extract rating
                rating_elem = metadata.select_one('.meta-title.flex-meta-rating .meta-value')
                updated_movie_dict['rating'] = (rating_elem.text.strip() if rating_elem else None)

                # Extract runtime
                runtime_elem = metadata.select_one('.meta-title.flex-meta-runtime .meta-value')
                updated_movie_dict['runtime'] = (runtime_elem.text.strip() if runtime_elem else None)

                # Extract genres
                genre_elem = metadata.select_one('.meta-title:has(.meta-label:-soup-contains("Genre:")) .meta-value')
                updated_movie_dict['genres'] = ([g.strip().lower() for g in genre_elem.text.split(',')] if genre_elem else None)
                updated_movie_dict['animation'] = (1 if "animation" in updated_movie_dict['genres'] else 0)

            # Extract credits
            people = soup.select('ul.credits li')
            if people:
                for credit in people:
                    title = credit.select_one('.credits-title')
                    value = credit.select_one('.credits-value')

                    if title and value:
                        credit_type = title.text.strip().lower()
                        credit_values = [name.strip() for name in value.text.split(',')]

                        if 'directed by' in credit_type:
                            updated_movie_dict['directors'] = credit_values
                        elif 'written by' in credit_type:
                            updated_movie_dict['writers'] = credit_values
                        elif 'cast' in credit_type:
                            updated_movie_dict['cast'] = credit_values

            time.sleep(0.2)  # Be nice to Disney's servers
            return updated_movie_dict

        except Exception as e:
            self.logger.error(f"Error processing {movie_dict.get('title')}: {str(e)}")

    def run(self):
        """
        Execution / run function to run the scraper and save each JSON file to directory.
        """
        # Get all movies
        movies = self.get_all_movies()

        # Process each movie
        self.logger.info("Processing individual movies...")
        for movie in tqdm(movies, desc="Processing movies"):
            # Process the movie
            self.process_movie(movie)
        self.logger.info(f"Completed scraping of movie titles.")

