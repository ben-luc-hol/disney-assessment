from transformers import T5Tokenizer, T5EncoderModel
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import json
import logging
import faiss
import pickle
from tqdm.notebook import tqdm


class MovieEmbedder:
    """
    Creates embeddings for movie data using T5.
    """
    def __init__(self,
                 logger: logging.Logger,
                 preprocessed_dir: Path,
                 embeddings_dir: Path,
                 model_name: str = 't5-base'):
        self.logger = logger
        self.preprocessed_dir = preprocessed_dir
        self.embeddings_dir = embeddings_dir
        self.embeddings_dir.mkdir(exist_ok=True)

        # Initialize T5
        self.logger.info(f"Loading {model_name} model and tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)

        # MPS if available ...
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("Using Apple Metal GPU acceleration")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU for computations")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Mappings storage
        self.index_to_slug = {}
        self.slug_to_index = {}

        self.batch_size = 32


    def create_movie_text(self, movie: Dict) -> str:
        """
        Create a rich text representation of a movie for embedding.
        """
        text_parts = [
            f"Title: {movie['title']}",
            f"Description: {movie['description']}",
            f"Year: {movie['year']}" if movie.get('year') else "",
            f"Directors: {', '.join(movie['directors'])}" if movie.get('directors') else "",
            f"Genres: {', '.join(movie['genres'])}" if movie.get('genres') else "",
            f"Cast: {', '.join(movie['cast'][:5])}" if movie.get('cast') else "",  # Limit to first 5 cast members
            f"Rating: {movie['rating']}" if movie.get('rating') else "",
            f"Type: {movie['type']}" if movie.get('type') else ""
        ]
        return " ".join(filter(None, text_parts))

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate T5 embedding for a piece of text.
        """
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(
                text,
                max_length=512,  # T5's max length
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling over the sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)

            if self.device == torch.device("mps"):
                embeddings = embeddings.to("cpu")
            return embeddings.numpy()

        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build and return a FAISS index optimized for CPU search.
        """
        dimension = embeddings.shape[1]

        # Use IVF index for better CPU performance
        nlist = min(int(np.sqrt(len(embeddings))), 100)  # number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        # Need to train the index
        if len(embeddings) < nlist:
            self.logger.warning(f"Not enough embeddings ({len(embeddings)}) for {nlist} clusters, adjusting...")
            nlist = max(1, len(embeddings) // 2)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        index.train(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))

        # Set number of probes (trade-off between speed and accuracy)
        index.nprobe = min(20, nlist)

        return index

    def process_embeddings(self):
        """
        Process all movies, generate embeddings, and build FAISS index.
        """
        self.logger.info("Starting embedding generation and indexing")

        movie_files = list(self.preprocessed_dir.glob('*.json'))
        self.logger.info(f"Found {len(movie_files)} movies to process")

        all_embeddings = []
        current_index = 0

        # Process in optimized batches
        for i in tqdm(range(0, len(movie_files), self.batch_size),
                      desc="Generating embeddings"):
            batch_files = movie_files[i:i + self.batch_size]
            batch_texts = []
            batch_slugs = []

            # Prepare batch
            for json_file in batch_files:
                try:
                    with open(json_file, 'r') as f:
                        movie = json.load(f)
                    movie_text = self.create_movie_text(movie)
                    batch_texts.append(movie_text)
                    batch_slugs.append(movie['slug'])

                    # Map index to slug
                    self.index_to_slug[current_index] = movie['slug']
                    self.slug_to_index[movie['slug']] = current_index
                    current_index += 1

                except Exception as e:
                    self.logger.error(f"Error reading {json_file.name}: {e}")
                    continue

            # Generate embeddings for batch
            try:
                embeddings = self.generate_text_embedding(batch_texts)
                all_embeddings.append(embeddings)

            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                continue

        # Combine all embeddings
        self.logger.info("Combining embeddings...")
        all_embeddings = np.vstack(all_embeddings)

        # Build FAISS index
        self.logger.info("Building FAISS index...")
        index = self.build_faiss_index(all_embeddings)

        # Save everything
        self.save_artifacts(index, all_embeddings)

        self.logger.info("Completed embedding generation and indexing")
        return index, all_embeddings

    def save_artifacts(self, index: faiss.Index, embeddings: np.ndarray):
        """
        Save FAISS index, embeddings, and mapping dictionaries.
        """
        # Save FAISS index
        faiss.write_index(index, str(self.embeddings_dir / 'movie_embeddings.index'))

        # Save embeddings
        np.save(self.embeddings_dir / 'embeddings.npy', embeddings)

        # Save mapping dictionaries
        with open(self.embeddings_dir / 'index_mapping.pkl', 'wb') as f:
            pickle.dump({
                'index_to_slug': self.index_to_slug,
                'slug_to_index': self.slug_to_index
            }, f)

        self.logger.info(f"Saved artifacts to {self.embeddings_dir}")

    def find_similar_movies(self, query, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find k most similar movies to a query text.
        """
        # Generate embedding for query
        # Load index
        embedding = self.generate_text_embedding(query)

        index = faiss.read_index(str(self.embeddings_dir / 'movie_embeddings.index'))

        # Search
        D, I = index.search(embedding.astype(np.float32), k)

        # Convert indices to slugs with distances
        results = [(self.index_to_slug[idx], dist)
                   for idx, dist in zip(I[0], D[0])]

        return results

