"""
Utilities package for the movie recommender system.
Contains helper functions for data processing and common operations.
"""

from .data_processor import (
    load_and_clean_movies,
    load_and_clean_ratings,
    preprocess_genres,
    create_sample_data,
    validate_movie_data
)

__all__ = [
    'load_and_clean_movies',
    'load_and_clean_ratings', 
    'preprocess_genres',
    'create_sample_data',
    'validate_movie_data'
]
