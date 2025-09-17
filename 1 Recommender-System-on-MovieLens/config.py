import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"

# Model settings
MODEL_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.8,
    "stop_words": "english"
}

# Recommendation settings
RECOMMENDATION_CONFIG = {
    "default_num_recommendations": 8,
    "max_recommendations": 20,
    "min_similarity_threshold": 0.1
}

# API settings
API_CONFIG = {
    "title": "Content-Based Movie Recommender",
    "description": "A movie recommendation system using TF-IDF and NearestNeighbors",
    "version": "1.0.0",
    "host": "0.0.0.0",
    "port": 8000
}
