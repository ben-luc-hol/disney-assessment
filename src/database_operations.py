import sqlite3
from pathlib import Path
import json
from typing import Dict, List, Tuple
from tqdm.notebook import tqdm
from .project_utils import ProjectManager


class DatabaseOps:
    """
    Manages database operations for the Disney movie dataset.
    """
    def __init__(self, project_manager):
        """Initialize database connection and create tables if they don't exist."""
        self.project_root = project_manager.project_root
        self.logger = ProjectManager.setup_component_logging(root=self.project_root, name=self.__class__.__name__)
        self.db_path = project_manager.directories['processed'] / 'movie_data.db'
        self.initialize_database()
        self.logger.info(f"Database initialized at {self.db_path}")

    def connect(self):
        """Create a database connection."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise

    def initialize_database(self):
        """
        Create all necessary tables.
        """
        create_tables_sql = ("""
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            year INTEGER,
            release_date TEXT,
            description TEXT,
            slug TEXT UNIQUE,
            type TEXT,
            link TEXT,
            rating TEXT,
            runtime TEXT,
            animation INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS genres (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS movie_genres (
            movie_id INTEGER,
            genre_id INTEGER,
            PRIMARY KEY (movie_id, genre_id),
            FOREIGN KEY (movie_id) REFERENCES movies (id),
            FOREIGN KEY (genre_id) REFERENCES genres (id)
        );

        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS movie_credits (
            movie_id INTEGER,
            person_id INTEGER,
            role_type TEXT CHECK(role_type IN ('director', 'writer', 'cast')),
            PRIMARY KEY (movie_id, person_id, role_type),
            FOREIGN KEY (movie_id) REFERENCES movies (id),
            FOREIGN KEY (person_id) REFERENCES people (id)
        );
        """)
        try:
            with self.connect() as conn:
                conn.executescript(create_tables_sql)
                self.logger.info(f"Initialized database at {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    def insert_movie(self, movie_data: Dict) -> int:
        """
        Insert a movie and its related data into the database.
        Returns the movie_id of the inserted movie.
        """
        try:
            with self.connect() as conn:
                # Insert movie base data
                movie_insert_sql = """
                INSERT INTO movies (
                    title, year, release_date, description, slug,
                    type, link, rating, runtime, animation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                cursor = conn.cursor()
                cursor.execute(movie_insert_sql, (
                    movie_data.get('title'),
                    movie_data.get('year'),
                    movie_data.get('release_date'),
                    movie_data.get('description'),
                    movie_data.get('slug'),
                    movie_data.get('type'),
                    movie_data.get('link'),
                    movie_data.get('rating'),
                    movie_data.get('runtime'),
                    movie_data.get('animation', 0)
                ))
                movie_id = cursor.lastrowid

                # Insert genres
                if movie_data.get('genres'):
                    self._insert_genres(conn, movie_id, movie_data['genres'])

                # Insert credits
                for role_type in ['directors', 'writers', 'cast']:
                    if movie_data.get(role_type):
                        self._insert_credits(conn, movie_id, movie_data[role_type], role_type[:-1])

                return movie_id

        except sqlite3.Error as e:
            self.logger.error(f"Error inserting movie {movie_data.get('title')}: {e}")
            raise

    def _insert_genres(self, conn: sqlite3.Connection, movie_id: int, genres: List[str]):
        """
        Helper method to insert genres for a movie.
        """
        for genre in genres:
            # Insert genre if not exists
            conn.execute(
                "INSERT OR IGNORE INTO genres (name) VALUES (?)",
                (genre,)
            )

            # Get genre_id
            cursor = conn.execute(
                "SELECT id FROM genres WHERE name = ?",
                (genre,)
            )
            genre_id = cursor.fetchone()[0]

            # Link movie to genre
            conn.execute(
                "INSERT OR IGNORE INTO movie_genres (movie_id, genre_id) VALUES (?, ?)",
                (movie_id, genre_id)
            )

    def _insert_credits(self, conn: sqlite3.Connection, movie_id: int, people: List[str], role_type: str):
        """
        Helper method to insert credits (cast, directors, writers) for a movie.
        """
        for person in people:
            # Insert person if not exists
            conn.execute(
                "INSERT OR IGNORE INTO people (name) VALUES (?)",
                (person,)
            )

            # Get person_id
            cursor = conn.execute(
                "SELECT id FROM people WHERE name = ?",
                (person,)
            )
            person_id = cursor.fetchone()[0]

            # Link movie to person with role
            conn.execute(
                "INSERT OR IGNORE INTO movie_credits (movie_id, person_id, role_type) VALUES (?, ?, ?)",
                (movie_id, person_id, role_type)
            )

    def batch_insert_from_json_files(self, json_dir: Path) -> Tuple[int, int]:
        """
        Batch insert movies from JSON files in the specified directory.

        Returns:
            Tuple of (processed_count, failed_count)
        """
        processed = 0
        failed = 0

        self.logger.info(f"Starting batch insert from {json_dir}")

        for json_file in tqdm(json_dir.glob('*.json'), desc="Loading movies to database"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    movie_data = json.load(f)
                    self.insert_movie(movie_data)
                    processed += 1

            except Exception as e:
                self.logger.error(f"Error processing {json_file.name}: {e}")
                failed += 1

        # Just log final summary
        self.logger.info(f"Database loading completed: {processed} succeeded, {failed} failed")
        return processed, failed

    def get_movie_data(self, movie_id: int) -> Dict:
        """
        Retrieve complete movie data including genres and credits.
        """
        try:
            with self.connect() as conn:
                # Get movie base data
                cursor = conn.execute("""
                    SELECT * FROM movies WHERE id = ?
                """, (movie_id,))
                movie = dict(cursor.fetchone())

                # Get genres
                cursor = conn.execute("""
                    SELECT g.name
                    FROM genres g
                    JOIN movie_genres mg ON g.id = mg.genre_id
                    WHERE mg.movie_id = ?
                """, (movie_id,))
                movie['genres'] = [row[0] for row in cursor.fetchall()]

                # Get credits
                for role_type in ['director', 'writer', 'cast']:
                    cursor = conn.execute("""
                        SELECT p.name
                        FROM people p
                        JOIN movie_credits mc ON p.id = mc.person_id
                        WHERE mc.movie_id = ? AND mc.role_type = ?
                    """, (movie_id, role_type))
                    movie[f"{role_type}s"] = [row[0] for row in cursor.fetchall()]

                return movie
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving movie {movie_id}: {e}")
            raise

    def get_movie_by_slug(self, slug: str) -> Dict:
        """
        Retrieve complete movie data using slug.
        """
        try:
            with self.connect() as conn:
                # Get movie base data and id
                cursor = conn.execute("""
                    SELECT * FROM movies WHERE slug = ?
                """, (slug,))
                movie = dict(cursor.fetchone())
                movie_id = movie['id']

                # Get genres and credits using movie_id
                cursor = conn.execute("""
                    SELECT g.name
                    FROM genres g
                    JOIN movie_genres mg ON g.id = mg.genre_id
                    WHERE mg.movie_id = ?
                """, (movie_id,))
                movie['genres'] = [row[0] for row in cursor.fetchall()]

                # Get credits
                for role_type in ['director', 'writer', 'cast']:
                    cursor = conn.execute("""
                                        SELECT p.name
                                        FROM people p
                                        JOIN movie_credits mc ON p.id = mc.person_id
                                        WHERE mc.movie_id = ? AND mc.role_type = ?
                                    """, (movie_id, role_type))
                    movie[f"{role_type}s"] = [row[0] for row in cursor.fetchall()]

            return movie
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving movie with slug {slug}: {e}")
            raise

    def get_all_movies(self) -> List[Dict]:
        """
        Get all movies with their genres and credits.
        """
        try:
            with self.connect() as conn:
                cursor = conn.execute("SELECT id FROM movies")
                movie_ids = [row[0] for row in cursor.fetchall()]
                return [self.get_movie_data(movie_id) for movie_id in movie_ids]
        except sqlite3.Error as e:
            self.logger.error("Error retrieving all movies: {e}")
            raise
