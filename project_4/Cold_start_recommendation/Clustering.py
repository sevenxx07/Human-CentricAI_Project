import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from .Factorization_engine import Matrix_Factorization
from sklearn.metrics.pairwise import cosine_similarity
import os
from django.conf import settings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path_movies = os.path.join(BASE_DIR, 'data', 'ml_latest_small', 'movies.csv')
csv_path_ratings = os.path.join(BASE_DIR, 'data', 'ml_latest_small', 'ratings.csv')


class TrueHybridColdStartSelector:
    """
    True hybrid movie selection that combines both genre-based and latent feature clustering
    from the beginning to get optimal cold start movie selection.
    """

    def __init__(self, df_movies, df_ratings, V_matrix=None, movie_to_index_map=None):
        self.df_movies = df_movies
        self.df_ratings = df_ratings
        self.V_matrix = V_matrix
        self.movie_to_index_map = movie_to_index_map
        self.mlb = None

    def get_genre_features(self, movies_df):
        """Extract genre features for movies"""
        movies_copy = movies_df.copy()
        movies_copy['genre_list'] = movies_copy['genres'].str.split('|')

        if self.mlb is None:
            self.mlb = MultiLabelBinarizer()
            genre_features = self.mlb.fit_transform(movies_copy['genre_list'])
        else:
            genre_features = self.mlb.transform(movies_copy['genre_list'])

        return genre_features

    def get_latent_features(self, movies_df):
        """Extract latent features for movies that exist in V matrix"""
        if self.V_matrix is None or self.movie_to_index_map is None:
            return None, []

        movie_features = []
        valid_movie_indices = []

        for idx, row in movies_df.iterrows():
            movie_id = row['movieId']
            if movie_id in self.movie_to_index_map:
                matrix_idx = self.movie_to_index_map[movie_id]
                if matrix_idx < len(self.V_matrix):
                    movie_features.append(self.V_matrix[matrix_idx])
                    valid_movie_indices.append(idx)

        if len(movie_features) == 0:
            return None, []

        return np.array(movie_features), valid_movie_indices

    def calculate_movie_quality_scores(self, movies_df):
        """Calculate quality scores for movies based on rating statistics"""
        movie_ids = movies_df['movieId'].tolist()
        movie_ratings = self.df_ratings[self.df_ratings['movieId'].isin(movie_ids)]

        if movie_ratings.empty:
            # Return default scores if no ratings available
            return pd.Series(0.5, index=movies_df.index)

        # Calculate statistics
        stats = movie_ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'var']
        }).reset_index()
        stats.columns = ['movieId', 'rating_count', 'rating_mean', 'rating_variance']
        stats['rating_variance'] = stats['rating_variance'].fillna(0)

        # Create quality score
        if len(stats) > 1:
            # Normalize metrics
            count_norm = stats['rating_count'] / stats['rating_count'].max()
            var_norm = stats['rating_variance'] / max(stats['rating_variance'].max(), 1e-6)
            mean_norm = 1 - abs(stats['rating_mean'] - 3.0) / 2.5

            stats['quality_score'] = (
                    0.6 * count_norm +  # Popularity
                    0.3 * var_norm +  # Informativeness
                    0.1 * mean_norm  # Reasonable rating
            )
        else:
            stats['quality_score'] = 0.5

        # Map back to movie dataframe
        score_map = dict(zip(stats['movieId'], stats['quality_score']))
        return movies_df['movieId'].map(score_map).fillna(0.3)

    def hybrid_clustering(self, n_clusters=8, genre_weight=0.6, latent_weight=0.4):
        """
        Perform hybrid clustering combining genre and latent features

        Args:
            n_clusters: Number of clusters to create
            genre_weight: Weight for genre-based features (0-1)
            latent_weight: Weight for latent features (0-1)
        """
        print(f"Performing hybrid clustering (Genre: {genre_weight}, Latent: {latent_weight})...")

        # Start with all movies
        working_movies = self.df_movies.copy()

        # Get genre features for all movies
        genre_features = self.get_genre_features(working_movies)
        print(f"Genre features shape: {genre_features.shape}")

        # Get latent features (only for movies that have them)
        latent_features, valid_latent_indices = self.get_latent_features(working_movies)

        if latent_features is not None:
            print(f"Latent features shape: {latent_features.shape}")
            print(f"Movies with latent features: {len(valid_latent_indices)}/{len(working_movies)}")

            # Create combined feature matrix
            # For movies with latent features: combine genre + latent
            # For movies without latent features: use only genre (with padding)

            combined_features = []

            for idx, row in working_movies.iterrows():
                genre_feat = genre_features[idx]

                if idx in valid_latent_indices:
                    # Movie has latent features - combine both
                    latent_idx = valid_latent_indices.index(idx)
                    latent_feat = latent_features[latent_idx]

                    # Normalize and combine
                    genre_norm = genre_feat / (np.linalg.norm(genre_feat) + 1e-8)
                    latent_norm = latent_feat / (np.linalg.norm(latent_feat) + 1e-8)

                    combined_feat = np.concatenate([
                        genre_weight * genre_norm,
                        latent_weight * latent_norm
                    ])
                else:
                    # Movie only has genre features - pad with zeros for latent part
                    genre_norm = genre_feat / (np.linalg.norm(genre_feat) + 1e-8)
                    latent_dim = self.V_matrix.shape[1] if self.V_matrix is not None else 11

                    combined_feat = np.concatenate([
                        genre_weight * genre_norm,
                        np.zeros(latent_dim) * latent_weight
                    ])

                combined_features.append(combined_feat)

            features_for_clustering = np.array(combined_features)
            clustering_type = "hybrid"

        else:
            # No latent features available - use only genre
            print("No latent features available, using genre-only clustering")
            features_for_clustering = genre_features
            clustering_type = "genre_only"

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_for_clustering)

        # Add cluster information to movies
        working_movies['cluster'] = clusters
        working_movies['clustering_type'] = clustering_type

        return working_movies, kmeans

    def select_best_from_clusters(self, clustered_movies, n_movies_per_cluster=1, top_k_candidates=5):
        """
        Select representative movies from each cluster with randomness

        Args:
            clustered_movies: DataFrame with clustered movies
            n_movies_per_cluster: Number of movies to select per cluster
            top_k_candidates: Number of top quality movies to choose from randomly
        """

        # Calculate quality scores for all movies
        quality_scores = self.calculate_movie_quality_scores(clustered_movies)
        clustered_movies['quality_score'] = quality_scores

        selected_movies = []

        for cluster_id in sorted(clustered_movies['cluster'].unique()):
            cluster_movies = clustered_movies[clustered_movies['cluster'] == cluster_id]

            # Sort movies in cluster by quality score (descending)
            cluster_movies_sorted = cluster_movies.sort_values('quality_score', ascending=False)

            # Get top k candidates (or all movies if cluster has fewer than k movies)
            n_candidates = min(top_k_candidates, len(cluster_movies_sorted))
            top_candidates = cluster_movies_sorted.head(n_candidates)

            # Randomly select one movie from the top candidates
            selected_movie = top_candidates.sample(n=1, random_state=None).iloc[0]

            selected_movies.append({
                'movieId': int(selected_movie['movieId']),
                'title': selected_movie['title'],
                'genres': selected_movie['genres'],
                'cluster': int(cluster_id),
                'clustering_type': selected_movie['clustering_type'],
                'quality_score': float(selected_movie['quality_score']),
                'candidates_considered': n_candidates
            })

        # Sort by quality score (descending)
        selected_movies.sort(key=lambda x: x['quality_score'], reverse=True)

        return selected_movies

    def get_hybrid_cold_start_movies(self, n_clusters=8, genre_weight=0.6, latent_weight=0.4, top_k_candidates=5):
        """
        Main method to get cold start movies using true hybrid approach with randomness

        Args:
            n_clusters: Number of clusters to create
            genre_weight: Importance of genre similarity (0-1)
            latent_weight: Importance of latent feature similarity (0-1)
            top_k_candidates: Number of top movies to randomly choose from in each cluster
        """

        # Normalize weights
        total_weight = genre_weight + latent_weight
        if total_weight > 0:
            genre_weight = genre_weight / total_weight
            latent_weight = latent_weight / total_weight

        print("=" * 60)
        print("HYBRID COLD START MOVIE SELECTION (WITH RANDOMNESS)")
        print("=" * 60)
        print(f"Strategy: Select randomly from top {top_k_candidates} movies per cluster")

        # Perform hybrid clustering
        clustered_movies, kmeans_model = self.hybrid_clustering(
            n_clusters=n_clusters,
            genre_weight=genre_weight,
            latent_weight=latent_weight
        )

        # Select movies from each cluster with randomness
        selected_movies = self.select_best_from_clusters(clustered_movies, top_k_candidates=top_k_candidates)

        print(f"\nSelected {len(selected_movies)} movies from {n_clusters} clusters:")
        print("-" * 60)

        for i, movie in enumerate(selected_movies, 1):
            print(f"{i:2d}. [Cluster {movie['cluster']}] {movie['title']}")
            print(f"     Genres: {movie['genres']}")
            print(
                f"     Quality Score: {movie['quality_score']:.3f} (from {movie['candidates_considered']} candidates)")
            print()

        return selected_movies, clustered_movies

    def analyze_clustering_quality(self, clustered_movies):
        """Analyze the quality of the clustering"""
        analysis = {
            'clustering_type': clustered_movies['clustering_type'].iloc[0],
            'total_movies': len(clustered_movies),
            'num_clusters': clustered_movies['cluster'].nunique(),
            'cluster_sizes': clustered_movies['cluster'].value_counts().to_dict(),
            'cluster_details': []
        }

        # Analyze each cluster
        for cluster_id in sorted(clustered_movies['cluster'].unique()):
            cluster_movies = clustered_movies[clustered_movies['cluster'] == cluster_id]

            # Get dominant genres
            all_genres = []
            for genres in cluster_movies['genres']:
                all_genres.extend(genres.split('|'))

            genre_counts = pd.Series(all_genres).value_counts()
            dominant_genres = genre_counts.head(3).to_dict()

            # Sample movie titles
            sample_titles = cluster_movies['title'].head(5).tolist()

            cluster_detail = {
                'cluster_id': cluster_id,
                'size': len(cluster_movies),
                'dominant_genres': dominant_genres,
                'sample_movies': sample_titles
            }

            analysis['cluster_details'].append(cluster_detail)

        return analysis


# Integration functions
def create_movie_index_map(R_matrix):
    """Create mapping from movieId to matrix index"""
    return {int(movie_id): idx for idx, movie_id in enumerate(R_matrix.columns)}


def run_true_hybrid_cold_start(df_movies, df_ratings, mat_fac_model=None, R_matrix=None,
                               n_clusters=8, genre_weight=0.6, latent_weight=0.4, top_k_candidates=3):
    """
    Complete workflow for true hybrid cold start movie selection with randomness

    Args:
        df_movies: Movies dataframe
        df_ratings: Ratings dataframe
        mat_fac_model: Trained matrix factorization model (optional)
        R_matrix: Rating matrix (optional)
        n_clusters: Number of clusters to create
        genre_weight: Weight for genre features (0-1)
        latent_weight: Weight for latent features (0-1)
        top_k_candidates: Number of top movies to randomly choose from per cluster
    """

    # Initialize hybrid selector
    if mat_fac_model is not None and R_matrix is not None:
        # We have trained model - use both genre and latent features
        if hasattr(mat_fac_model, 'V'):
            V = mat_fac_model.V
        else:
            # Assuming V is the second return value from factorize
            U, V = mat_fac_model.U, mat_fac_model.V

        movie_to_index_map = create_movie_index_map(R_matrix)
        selector = TrueHybridColdStartSelector(df_movies, df_ratings, V, movie_to_index_map)
        print("Initialized with BOTH genre and latent features")
    else:
        # No trained model - use only genre features
        selector = TrueHybridColdStartSelector(df_movies, df_ratings)
        print("Initialized with ONLY genre features")

    # Get hybrid cold start movies with randomness
    selected_movies, clustered_movies = selector.get_hybrid_cold_start_movies(
        n_clusters=n_clusters,
        genre_weight=genre_weight,
        latent_weight=latent_weight,
        top_k_candidates=top_k_candidates
    )

    # Analyze clustering quality
    analysis = selector.analyze_clustering_quality(clustered_movies)

    # print("\n" + "=" * 60)
    # print("CLUSTERING ANALYSIS")
    # print("=" * 60)
    # print(f"Clustering Type: {analysis['clustering_type']}")
    # print(f"Total Movies: {analysis['total_movies']}")
    # print(f"Number of Clusters: {analysis['num_clusters']}")
    # print(f"Random Selection: Top {top_k_candidates} candidates per cluster")
    # print()

    # for cluster_detail in analysis['cluster_details']:
    #     print(f"Cluster {cluster_detail['cluster_id']} ({cluster_detail['size']} movies):")
    #     print(f"  Dominant Genres: {cluster_detail['dominant_genres']}")
    #     print(f"  Sample Movies: {', '.join(cluster_detail['sample_movies'][:3])}")
    #     print()

    return selected_movies, clustered_movies, selector, analysis

if __name__ == "__main__":
    # Load your data (you already have this)
    df_movies = pd.read_csv(csv_path_movies)
    df_ratings = pd.read_csv(csv_path_ratings)
    R_matrix = pd.read_csv("R_matrix.csv", index_col=0)

    # Train your model (you already have this)
    mat_fac_model = Matrix_Factorization(R_matrix)
    U, V = mat_fac_model.factorize(11)

    # Store U and V in the model object for easy access
    mat_fac_model.U = U
    mat_fac_model.V = V

    # Now use the TRUE hybrid cold start selector
    selected_movies, clustered_movies, selector, analysis = run_true_hybrid_cold_start(
        df_movies=df_movies,
        df_ratings=df_ratings,
        mat_fac_model=mat_fac_model,
        R_matrix=R_matrix,
        n_clusters=10,
        genre_weight=0.7,  # 70% importance to genre similarity
        latent_weight=0.3  # 30% importance to latent feature similarity
    )

# Model to return the selected movies 
def get_selected_cold_start_movies():
    df_movies = pd.read_csv(csv_path_movies)
    df_ratings = pd.read_csv(csv_path_ratings)
    R_matrix = pd.read_csv("R_matrix.csv", index_col=0)
            
    mat_fac_model = Matrix_Factorization(R_matrix)
    U, V = mat_fac_model.factorize(11)

    mat_fac_model.U = U
    mat_fac_model.V = V
    selected_movies, _, _, _ = run_true_hybrid_cold_start(            
        df_movies=df_movies,
        df_ratings=df_ratings,
        mat_fac_model=mat_fac_model,
        R_matrix=R_matrix,            
        n_clusters=10,
        genre_weight=0.7,  # 70% importance to genre similarity
        latent_weight=0.3  # 30% importance to latent feature similarity
        )

    return selected_movies