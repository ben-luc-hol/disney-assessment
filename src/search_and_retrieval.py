from typing import List, Tuple, Dict
import json
from datetime import datetime
from anthropic import Anthropic


class SearchAndRetrieval:
    """
    End-to-end semantic search:
    1. Convert query to embedding
    2. Find similar vectors in FAISS
    3. Retrieve full records from SQLite
    4. Return formatted results
    """
    def __init__(self, output_dir, logger, embedder, db_manager):
        self.claude = None
        self.logger = logger
        self.embedder = embedder
        self.db_manager = db_manager
        self.output_dir = output_dir

    def semantic_search(self, query: str, k:int = 5, min_score: float = None):

        query = query.lower()

        try:
            # Get similar movie slugs from FAISS
            self.logger.info(f"Executing semantic search for: '{query}'")

            similar_slugs = self.embedder.find_similar_movies(query, k)

            # Get full movie details from database
            results = []
            for slug, score in similar_slugs:
                movie = self.db_manager.get_movie_by_slug(slug)
                movie['similarity_score'] = float(score)
                results.append(movie)

            self.save_results(query, results)
            return results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise

    def save_results(self, query: str, results: List[Dict]):
        """
        Save search results to disk for analysis.
        query: The search query
        results: List of movie dictionaries with similarity scores
        """
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Format data for saving
            search_data = {
                'query': query,
                'timestamp': timestamp,
                'num_results': len(results),
                'results': results
            }

            # Save to JSON file
            output_file = self.output_dir / f"search_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(search_data, f, indent=2)

            self.logger.info(f"Saved search results to {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save search results: {e}")

    @staticmethod
    def format_search_results(self, results: List[Dict]) -> str:
        """
        Format search results for display.
        """
        formatted = []
        for movie in results:
            movie_info = [
                f"\nTitle: {movie['title']} ({movie.get('year', 'N/A')})",
                f"Similarity Score: {movie['similarity_score']:.2f}",
                f"Genres: {', '.join(movie.get('genres', []))}",
                f"Directors: {', '.join(movie.get('directors', []))}",
                f"Description: {movie.get('description', 'N/A')[:200]}..."
            ]
            formatted.append("\n".join(movie_info))

        return "\n---\n".join(formatted)

    def generate_movie_response(self, query: str, k: int = 5, model: str = "claude-3-sonnet-20240229") -> str:
        """
        RAG implementation for Disney movie queries and recommendations.

        Args:
            query: User query about Disney movies (can be questions, recommendation requests, etc.)
            k: Number of movies to include in context
            model: LLM model to use for generation

        Returns:
            Generated response based on retrieved movies
        """
        self.claude = Anthropic()  # NOTE: API KEY NEEDS TO BE RETRIEVABLE AS AN ENV VARIABLE

        try:
            self.logger.info(f"Generating movie response for: '{query}'")

            # Get relevant movies
            movies = self.semantic_search(query, k=k)
            if not movies:
                return "No relevant movies found for your query."

            # Format context for the LLM
            context_parts = []
            for movie in movies:
                context = {
                    "title": movie['title'],
                    "year": movie.get('year', 'Unknown'),
                    "description": movie.get('description', ''),
                    "genres": ', '.join(movie.get('genres', [])),
                    "directors": ', '.join(movie.get('directors', [])),
                    "cast": ', '.join(movie.get('cast', [])[:5])  # Top 5 cast members
                }
                context_parts.append(context)

            # Create prompts for the LLM
            system_prompt, user_prompt = self._create_rag_prompt(query, context_parts)

            # Generate response using the LLM
            response = self._generate_llm_response(system_prompt, user_prompt, model)

            return response

        except Exception as e:
            self.logger.error(f"Movie response generation failed: {e}")
            raise

    def _create_rag_prompt(self, query: str, context_parts: List[Dict]) -> str:
        """Create prompt for the LLM using retrieved context."""
        context_text = "\n\n".join([
            f"Movie: {ctx['title']} ({ctx['year']})\n"
            f"Genre: {ctx['genres']}\n"
            f"Directors: {ctx['directors']}\n"
            f"Cast: {ctx['cast']}\n"
            f"Description: {ctx['description']}"
            for ctx in context_parts
        ])
        system_prompt = (f"""
            You are an AI assistant whose job is to recommend, summarize, and otherwise provide information about feature-length movies by Disney.
            You are powered by a data retrieval system that allows you to access a database that informs your answers to the user.
            
            The following user prompt yielded the following results from our database:
            
            <MOVIE DATA>
            {context_text}
            </MOVIE DATA>
            
            Please summarize, provide information , and/or recommend films to the user, based on this information.
            
                Please provide a detailed response that:
                 1. Directly addresses the question
                 2. References specific movies and their relevant details
                 3. Explains why these movies are relevant to the user's question
                 4. Includes interesting connections or patterns among the movies.
        """)
        
        user_prompt = f"""Based on the provided data and instructions, answer the following user question: "{query}"
        
        """
        return system_prompt, user_prompt

    def _generate_llm_response(self, system_prompt: str, user_prompt: str, model: str) -> str:
        """
        Generate response using Anthropic/Claude
        """
        response = self.claude.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.4,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        return response.content[0].text