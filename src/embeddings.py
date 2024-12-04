from transformers import T5Tokenizer, T5EncoderModel
import torch
from typing import List, Dict, Union, Tuple
from pathlib import Path
import numpy as np
import json
import logging
import faiss
from .project_utils import ProjectManager
import pickle
from tqdm.notebook import tqdm


class MovieEmbedder:
    """
    Creates embeddings for movie data using T5.
    """
    def __init__(self, project_manager, model_name: str = 't5-base'):
        self.logger = ProjectManager.setup_component_logging(root=project_manager.project_root, name=self.__class__.__name__)
        self.embeddings_dir = project_manager.directories['processed'] / 'embeddings'
        self.embeddings_dir.mkdir(exist_ok=True)
        self.project_manager = project_manager

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

    def generate_text_embedding(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate T5 embedding for a piece of text or list of texts.
        """
        try:
            if isinstance(texts, str):
                texts = [texts]

            print(f"Processing {len(texts)} texts for embedding")
            inputs = self.tokenizer(
                texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            print("Running model inference...")
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            print("Converting to numpy...")
            if self.device == torch.device("mps"):
                embeddings = embeddings.to("cpu")
            return embeddings.numpy()

        except Exception as e:
            print(f"Error in generate_text_embedding: {e}")
            self.logger.error(f"Error generating embedding: {e}")
            raise

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build and return a FAISS index optimized for CPU search.
        """
        self.logger.info("Starting FAISS index build...")
        dimension = embeddings.shape[1]

        # Use IVF index for better CPU performance
        nlist = min(int(np.sqrt(len(embeddings))), 100)  # number of clusters
        self.logger.info(f"Creating index with {nlist} clusters...")

        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        #  train the index
        if len(embeddings) < nlist:
            self.logger.warning(f"Not enough embeddings ({len(embeddings)}) for {nlist} clusters, adjusting...")
            nlist = max(1, len(embeddings) // 2)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        self.logger.info("Training FAISS index...")
        index.train(embeddings.astype(np.float32))

        self.logger.info("Adding vectors to index...")
        index.add(embeddings.astype(np.float32))

        # Set number of probes (trade-off between speed and accuracy)
        index.nprobe = min(20, nlist)

        self.logger.info("FAISS index build complete")
        return index

    # In save_artifacts:
    def save_artifacts(self, index: faiss.Index, embeddings: np.ndarray):
        """
        Save FAISS index, embeddings, and mapping dictionaries.
        """
        self.logger.info("Starting to save artifacts...")

        # Save FAISS index
        self.logger.info("Saving FAISS index...")
        faiss.write_index(index, str(self.embeddings_dir / 'movie_embeddings.index'))

        # Save embeddings
        self.logger.info("Saving embeddings array...")
        np.save(self.embeddings_dir / 'embeddings.npy', embeddings)

        # Save mapping dictionaries
        self.logger.info("Saving index mappings...")
        with open(self.embeddings_dir / 'index_mapping.pkl', 'wb') as f:
            pickle.dump({
                'index_to_slug': self.index_to_slug,
                'slug_to_index': self.slug_to_index
            }, f)

        self.logger.info("All artifacts saved successfully")

    def process_embeddings(self, movies: List[Dict], use_parallel: bool = False):
        """
        Process movies, generate embeddings, and build FAISS index.
        """
        print("Starting embedding generation and indexing")
        self.logger.info("Starting embedding generation and indexing")

        # Process sequentially
        print(f"Creating text representations for {len(movies)} movies...")
        movie_texts = [self.create_movie_text(movie) for movie in tqdm(movies, desc="Generating movie texts")]

        print("Setting up slug mappings...")
        # slug mappings
        for idx, movie in enumerate(movies):
            self.index_to_slug[idx] = movie['slug']
            self.slug_to_index[movie['slug']] = idx

        # Process embeddings in batches
        print(f"Generating embeddings in batches of {self.batch_size}...")
        all_embeddings = []
        total_batches = (len(movie_texts) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(movie_texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = movie_texts[i:i + self.batch_size]
            print(f"Starting batch {i // self.batch_size + 1}/{total_batches}")
            print(f"Batch size: {len(batch_texts)}")
            try:
                embeddings = self.generate_text_embedding(batch_texts)
                all_embeddings.append(embeddings)
                print(f"Completed batch {i // self.batch_size + 1}/{total_batches}")
            except Exception as e:
                print(f"Error processing batch {i // self.batch_size + 1}: {e}")
                raise

        print("\nAll batches completed. Starting post-processing...")
        print(f"Number of embedding batches: {len(all_embeddings)}")
        for i, emb in enumerate(all_embeddings):
            print(f"Batch {i + 1} shape: {emb.shape}")

        print("\nStarting vstack operation...")
        try:
            all_embeddings = np.vstack(all_embeddings)
            print(f"Vstack completed. Final shape: {all_embeddings.shape}")
        except Exception as e:
            print(f"Error during vstack: {e}")
            raise

        print("\nStarting FAISS index build...")

        try:
            index = self.build_faiss_index(all_embeddings)
            print("FAISS index built successfully")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            raise

        print("Starting to save artifacts...")
        try:
            self.save_artifacts(index, all_embeddings)
            print("Artifacts saved successfully")
        except Exception as e:
            print(f"Error saving artifacts: {e}")
            raise

        print("Done")
        return index, all_embeddings

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

