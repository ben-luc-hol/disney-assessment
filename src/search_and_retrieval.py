from typing import List, Tuple, Dict
import json
from datetime import datetime

class SearchAndRetrieval:
    """
    End-to-end semantic search:
    1. Convert query to embedding
    2. Find similar vectors in FAISS
    3. Retrieve full records from SQLite
    4. Return formatted results
    """
    def __init__(self, output_dir, logger, embedder, db_manager):
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

        Args:
            query: The search query
            results: List of movie dictionaries with similarity scores
        """
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Create results directory if it doesn't exist
            results_dir = self.project_root / 'results' / 'searches'
            results_dir.mkdir(parents=True, exist_ok=True)

            # Format data for saving
            search_data = {
                'query': query,
                'timestamp': timestamp,
                'num_results': len(results),
                'results': results
            }

            # Save to JSON file
            output_file = results_dir / f"search_{timestamp}.json"
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


    def generate_movie_summary(self,
                               query: str,
                               k: int = 3,
                               model: str = "gpt-3.5-turbo") -> str:
        """
        RAG implementation for generating summaries about Disney movies.

        Args:
            query: User query about Disney movies
            k: Number of movies to include in context
            model: LLM model to use for generation

        Returns:
            Generated response based on retrieved movies
        """
        try:
            self.logger.info(f"Generating movie summary for query: '{query}'")

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

            # Create prompt for the LLM
            prompt = self._create_rag_prompt(query, context_parts)

            # Generate response using the LLM
            response = self._generate_llm_response(prompt, model)

            self.logger.info("Successfully generated movie summary")
            return response

        except Exception as e:
            self.logger.error(f"Movie summary generation failed: {e}")
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

        prompt = f"""Based on the following Disney movies, answer this question: "{query}"
    
    Retrieved Movie Information:
    {context_text}
    
    Please provide a detailed response that:
    1. Directly addresses the question
    2. References specific movies and their relevant details
    3. Explains why these movies are relevant to the query
    4. Includes interesting connections or patterns among the movies
    
    Response:"""

        return prompt


    def _generate_llm_response(self, prompt: str, model: str) -> str:
        """
        Generate response using specified LLM.
        This example uses OpenAI, but could be adapted for other LLMs.
        """
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable Disney movie expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise