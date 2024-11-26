# Disney+ Movie Data Pipeline and Search System 🎬

A  data pipeline that scrapes Disney movie data, processes it, generates embeddings, and enables semantic search with RAG capabilities.

## 🎯 Overview

This project implements an end-to-end data pipeline for Disney+ movie data that includes:

- 📥 Data collection from Disney's movie listings
- 🔄 Data preprocessing and cleaning
- 🧮 Vector embeddings generation using T5
- 💾 Efficient vector storage using FAISS
- 🔍 Semantic search capabilities
- 🤖 Retrieval Augmented Generation (RAG) implementation

## 🏗️ Architecture

```
project/
├── src/
│   ├── project_utils.py     # Logging, parallel processing utilities
│   ├── data_ingestion.py    # Movie data scraping
│   ├── data_preprocessing.py # Data cleaning and transformation
│   ├── database_operations.py # SQLite database operations
│   ├── embeddings.py        # T5 embeddings generation
│   └── search_and_retrieval.py # Search and RAG implementation
├── data/
│   ├── raw/                 # Raw JSON movie data
│   └── processed/           # Processed data and embeddings
├── logs/                    # Pipeline execution logs
└── notebooks/              
    └── pipeline_demo.ipynb  # Demo notebook
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/disney-movie-pipeline.git
cd disney-movie-pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Pipeline Execution

```python
from src import DisneyETL

# Initialize pipeline
etl = DisneyETL()

# Run data collection
etl.collect_data()

# Process and load to database
etl.load_to_sqlite()

# Generate embeddings
etl.vectorize_data()
```

### Semantic Search

```python
# Search for similar movies
results = etl.movie_search("animated movies about family and adventure", k=5)
print(results)
```

## ✨ Features

- **🔄 Robust Data Collection**: Scrapes comprehensive movie data including titles, descriptions, cast, genres, and more
- **⚡️ Efficient Processing**: Implements parallel processing for data transformation
- **🔍 Advanced Search**: Semantic search using T5 embeddings and FAISS
- **📈 Scalable Design**: Modular architecture with proper logging and error handling
- **🚀 Performance Optimized**: 
  - Batch processing for embeddings generation
  - FAISS indexing for fast similarity search
  - GPU acceleration support (Apple Metal)

## 🛠️ Technical Details

- **Embedding Model**: T5-base for generating text embeddings
- **Vector Storage**: FAISS IVF index for efficient similarity search
- **Database**: SQLite for structured data storage
- **Text Processing**: Custom preprocessing pipeline for cleaning and normalizing text data
- **Logging**: Comprehensive logging system with both file and optional console outputs

## 📊 Performance

- Processes ~700 movies in under 5 minutes
- Average query time < 100ms
- Embeddings generation optimized with batch processing
- FAISS index provides sub-linear search complexity

## 🔜 Future Improvements

- Implement incremental updates
- Add more advanced RAG capabilities
- Expand the data and provide a richer dataset
- Enhance search with filtering options
- Add support for more embedding models
- Implement caching for frequently accessed data

