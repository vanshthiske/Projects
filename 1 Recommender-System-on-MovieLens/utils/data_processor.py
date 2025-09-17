"""
Data processing utilities for the movie recommender system.
Handles data loading, cleaning, preprocessing, and validation.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_clean_movies(file_path: str) -> pd.DataFrame:
    """
    Load and clean the movies dataset.
    
    Args:
        file_path (str): Path to the movies CSV file
        
    Returns:
        pd.DataFrame: Cleaned movies dataframe
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the required columns are missing
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Movies file not found at: {file_path}")
        
        # Load the data
        logger.info(f"Loading movies data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['movieId', 'title', 'genres']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Loaded {len(df)} movies initially")
        
        # Clean the data
        df = preprocess_genres(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['movieId']).reset_index(drop=True)
        
        logger.info(f"After cleaning: {len(df)} movies remaining")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading movies data: {e}")
        raise

def load_and_clean_ratings(file_path: str) -> pd.DataFrame:
    """
    Load and clean the ratings dataset.
    
    Args:
        file_path (str): Path to the ratings CSV file
        
    Returns:
        pd.DataFrame: Cleaned ratings dataframe
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Ratings file not found at: {file_path}")
            return pd.DataFrame()
        
        logger.info(f"Loading ratings data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['userId', 'movieId', 'rating']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in ratings: {missing_columns}")
        
        # Clean ratings data
        df = df.dropna(subset=['userId', 'movieId', 'rating'])
        df = df[(df['rating'] >= 0.5) & (df['rating'] <= 5.0)]  # Valid rating range
        
        logger.info(f"Loaded and cleaned {len(df)} ratings")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading ratings data: {e}")
        raise

def preprocess_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the genres column.
    
    Args:
        df (pd.DataFrame): Movies dataframe with 'genres' column
        
    Returns:
        pd.DataFrame: Dataframe with cleaned genres
    """
    df = df.copy()
    
    # Handle missing values
    df['genres'] = df['genres'].fillna('')
    
    # Clean genres text
    df['genres_clean'] = (
        df['genres']
        .str.replace('(no genres listed)', '', regex=False)
        .str.replace('|', ' ', regex=False)  # Replace pipe with space
        .str.replace('-', ' ', regex=False)  # Replace hyphens with space
        .str.strip()  # Remove leading/trailing spaces
        .str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with single space
    )
    
    # Remove movies without valid genres
    df = df[
        (df['genres_clean'].str.len() > 0) & 
        (df['genres_clean'] != '') &
        (df['genres_clean'] != ' ')
    ].reset_index(drop=True)
    
    return df

def validate_movie_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the movie dataset for common issues.
    
    Args:
        df (pd.DataFrame): Movies dataframe to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for required columns
    required_columns = ['movieId', 'title', 'genres_clean']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing columns: {missing_columns}")
    
    # Check for empty dataframe
    if len(df) == 0:
        issues.append("Dataset is empty")
    
    # Check for missing titles
    if df['title'].isna().any():
        issues.append(f"Found {df['title'].isna().sum()} movies with missing titles")
    
    # Check for duplicate movie IDs
    duplicate_ids = df['movieId'].duplicated().sum()
    if duplicate_ids > 0:
        issues.append(f"Found {duplicate_ids} duplicate movie IDs")
    
    # Check genres
    empty_genres = df['genres_clean'].str.len() == 0
    if empty_genres.any():
        issues.append(f"Found {empty_genres.sum()} movies with empty genres")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues

def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create sample movie and ratings data for testing purposes.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (movies_df, ratings_df)
    """
    # Sample movies data
    movies_data = {
        'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'title': [
            'Toy Story (1995)', 
            'Jumanji (1995)', 
            'Grumpier Old Men (1995)',
            'Waiting to Exhale (1995)', 
            'Father of the Bride Part II (1995)',
            'Heat (1995)', 
            'Sabrina (1995)', 
            'Tom and Huck (1995)',
            'Sudden Death (1995)', 
            'GoldenEye (1995)',
            'The Lion King (1994)',
            'Forrest Gump (1994)',
            'Pulp Fiction (1994)',
            'The Matrix (1999)',
            'Titanic (1997)'
        ],
        'genres': [
            'Adventure|Animation|Children|Comedy|Fantasy',
            'Adventure|Children|Fantasy',
            'Comedy|Romance',
            'Comedy|Drama|Romance',
            'Comedy',
            'Action|Crime|Thriller',
            'Comedy|Romance',
            'Adventure|Children',
            'Action',
            'Action|Adventure|Thriller',
            'Animation|Children|Drama|Musical',
            'Comedy|Drama|Romance',
            'Crime|Drama',
            'Action|Sci-Fi|Thriller',
            'Drama|Romance'
        ]
    }
    
    # Sample ratings data
    ratings_data = {
        'userId': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
        'movieId': [1, 2, 3, 11, 1, 4, 12, 2, 6, 13, 3, 14, 1, 5, 15],
        'rating': [5.0, 4.0, 3.0, 4.5, 4.5, 3.5, 4.0, 4.5, 2.0, 4.5, 3.0, 4.0, 4.0, 3.0, 4.5],
        'timestamp': [964982703, 964981247, 964982224, 964982931, 964982400, 964982346, 
                     964982503, 964982148, 964982703, 964982620, 964982441, 964982901,
                     964982765, 964982588, 964982653]
    }
    
    movies_df = pd.DataFrame(movies_data)
    ratings_df = pd.DataFrame(ratings_data)
    
    # Clean the movies data
    movies_df = preprocess_genres(movies_df)
    
    logger.info(f"Created sample data: {len(movies_df)} movies, {len(ratings_df)} ratings")
    
    return movies_df, ratings_df

def get_genre_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics about genres in the dataset.
    
    Args:
        df (pd.DataFrame): Movies dataframe with 'genres_clean' column
        
    Returns:
        pd.DataFrame: Genre statistics
    """
    # Split genres and count occurrences
    all_genres = []
    for genres_str in df['genres_clean']:
        genres = genres_str.split()
        all_genres.extend(genres)
    
    # Count and create statistics
    genre_counts = pd.Series(all_genres).value_counts()
    
    stats_df = pd.DataFrame({
        'genre': genre_counts.index,
        'count': genre_counts.values,
        'percentage': (genre_counts.values / len(df) * 100).round(2)
    })
    
    return stats_df

def filter_popular_movies(df: pd.DataFrame, ratings_df: pd.DataFrame, 
                         min_ratings: int = 10) -> pd.DataFrame:
    """
    Filter movies based on minimum number of ratings.
    
    Args:
        df (pd.DataFrame): Movies dataframe
        ratings_df (pd.DataFrame): Ratings dataframe
        min_ratings (int): Minimum number of ratings required
        
    Returns:
        pd.DataFrame: Filtered movies dataframe
    """
    if ratings_df.empty:
        logger.warning("No ratings data provided, returning all movies")
        return df
    
    # Count ratings per movie
    rating_counts = ratings_df['movieId'].value_counts()
    popular_movies = rating_counts[rating_counts >= min_ratings].index
    
    # Filter movies
    filtered_df = df[df['movieId'].isin(popular_movies)].reset_index(drop=True)
    
    logger.info(f"Filtered to {len(filtered_df)} popular movies (min {min_ratings} ratings)")
    
    return filtered_df

def export_processed_data(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, 
                         output_dir: str = "data/processed") -> None:
    """
    Export processed data to CSV files.
    
    Args:
        movies_df (pd.DataFrame): Processed movies dataframe
        ratings_df (pd.DataFrame): Processed ratings dataframe
        output_dir (str): Directory to save processed files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export movies
    movies_path = os.path.join(output_dir, "movies_processed.csv")
    movies_df.to_csv(movies_path, index=False)
    logger.info(f"Exported processed movies to: {movies_path}")
    
    # Export ratings if available
    if not ratings_df.empty:
        ratings_path = os.path.join(output_dir, "ratings_processed.csv")
        ratings_df.to_csv(ratings_path, index=False)
