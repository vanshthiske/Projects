import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.knn_model = None
        self.title_to_index = None
        self.tfidf_vectorizer = None
        
    def load_data(self, movies_path):
        """Load and preprocess movie data"""
        try:
            self.movies_df = pd.read_csv(movies_path)
            
            # Handle missing genres
            self.movies_df['genres'] = self.movies_df['genres'].fillna('')
            
            # Clean genres
            self.movies_df['genres_clean'] = (
                self.movies_df['genres']
                .str.replace('(no genres listed)', '', regex=False)
                .str.replace('|', ' ', regex=False)
                .str.strip()
            )
            
            # Remove movies without genres
            self.movies_df = self.movies_df[
                (self.movies_df['genres_clean'].str.len() > 0) & 
                (self.movies_df['genres_clean'] != '')
            ].reset_index(drop=True)
            
            print(f"✅ Loaded {len(self.movies_df)} movies")
            
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def build_model(self):
        """Build TF-IDF matrix and NearestNeighbors model"""
        if self.movies_df is None:
            raise Exception("No data loaded. Call load_data() first.")
        
        try:
            # Create TF-IDF vectors
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                max_features=5000,
                ngram_range=(1, 2)  # Include bigrams for better matching
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.movies_df['genres_clean']
            )
            
            # Build KNN model
            self.knn_model = NearestNeighbors(
                n_neighbors=min(50, len(self.movies_df)),
                metric='cosine',
                algorithm='brute'
            )
            self.knn_model.fit(self.tfidf_matrix)
            
            # Create title mapping
            self.title_to_index = pd.Series(
                self.movies_df.index, 
                index=self.movies_df['title']
            ).drop_duplicates()
            
            print(f"✅ Model built with {self.tfidf_matrix.shape[1]} features")
            
        except Exception as e:
            raise Exception(f"Error building model: {e}")
    
    def get_recommendations(self, movie_title, num_recommendations=8):
        """Get movie recommendations for a given title"""
        if movie_title not in self.title_to_index:
            return None
        
        try:
            # Get movie index
            movie_idx = self.title_to_index[movie_title]
            movie_vector = self.tfidf_matrix[movie_idx]
            
            # Find similar movies
            distances, indices = self.knn_model.kneighbors(
                movie_vector, 
                n_neighbors=num_recommendations + 1
            )
            
            # Get recommendations (exclude the input movie)
            recommended_indices = indices.flatten()[1:]
            similarity_scores = 1 - distances.flatten()[1:]  # Convert distance to similarity
            
            recommendations = []
            for idx, score in zip(recommended_indices, similarity_scores):
                movie_data = self.movies_df.iloc[idx]
                recommendations.append({
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'similarity_score': round(score * 100, 1),
                    'movieId': movie_data.get('movieId', idx)
                })
            
            return recommendations
            
        except Exception as e:
            raise Exception(f"Error getting recommendations: {e}")
    
    def search_movies(self, query, limit=10):
        """Search for movies matching query"""
        if self.movies_df is None:
            return []
        
        query_lower = query.lower()
        matches = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(query_lower, na=False)
        ]['title'].head(limit).tolist()
        
        return matches
    
    def get_all_movie_titles(self):
        """Get all movie titles for dropdown"""
        if self.movies_df is None:
            return []
        return sorted(self.movies_df['title'].tolist())
