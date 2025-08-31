from .Clustering import get_selected_cold_start_movies
from .Factorization_engine import get_R_U_V
from .feature_interpretations import feature_dict, feature_characteristics
import pandas as pd
import os
import numpy as np
import random
from numpy.linalg import norm


def load_movie_data():
    """ Loading movie data and returning a mapping from 
    movieId to movie title"""

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    csv_path_movies = os.path.join(BASE_DIR, 'data', 'ml_latest_small', 'movies.csv')
    df_movies = pd.read_csv(csv_path_movies)

    # Id to title dictionary
    movieId_to_title = dict(zip(df_movies['movieId'], df_movies['title']))
    return movieId_to_title


# ---------- Cold start ---------- #

def get_initial_movies(n=5):
    """Picking n diverse cold start movies to present 
    to the user in the initial stage"""

    selected_movies = get_selected_cold_start_movies()  # Delegates to the clustering (one movie per cluster)
    initial_movies = random.sample(selected_movies, n)  # Randomly sampling the clustered movies
    print(f"Initial movies:", initial_movies)
    return initial_movies


class ColdStart:
    """Estimating and updating the user's latent vector from a few ratings."""

    def __init__(self, V_matrix, R_matrix, lambd=0.001):
        self.V = V_matrix  # Latent feature matrix for movies
        self.K = V_matrix.shape[1]  # Nr of latent features
        self.R = R_matrix  # Original user-movie rating matrix
        # Ensure movie ids are ints
        self.R.columns = self.R.columns.astype(int)
        self.lambd = lambd  # Regularization parameter

    def update_user_vector(self, user_ratings):
        """Computing the initial user vector U_i from the first ratings using 
        reguralized least squares (eq. 2 in the instructions)"""

        rated_ids = list(user_ratings.keys())

        # Keep onl ids that exist as columns in R
        found_ids = [i for i in rated_ids if i in self.R.columns]
        if found_ids != rated_ids:
            not_found = set(rated_ids) - set(found_ids)
            print(f"ValueError: not found ids {not_found}")
            return np.zeros(self.K)

        # Building arrays containing ratings and the the V_rated matrix
        ratings = np.array(list(user_ratings.values()), dtype=np.float64)
        rated_indices = [self.R.columns.get_loc(i) for i in rated_ids if i in self.R.columns]
        V_rated = self.V[rated_indices, :]

        # Regularized least squares solution:
        A = V_rated.T @ V_rated + self.lambd * np.eye(self.K)
        b = V_rated.T @ ratings
        U_i = np.linalg.solve(A, b)
        return U_i


# ------------- Similarity utils -------------#
def cosine_similarity(a, b):
    """ Comparing (cosine) similarity between two vectors"""
    return np.dot(a, b) / (norm(a) * norm(b)) if norm(a) > 0 and norm(b) > 0 else 0


# ---------------- Explaination ---------------#

def explain_impact(current_user_ratings, movie_to_rate_id, V, R, movieId_to_title, top_k=1, feature_dict=feature_dict,
                   feature_information=feature_characteristics):
    """
    For a candidate movie to rate, all possible rating (1-5) are simulated and explained how
    each rating would affect the user's latent profile and the next top movie is selected.
    The top k latent features whose absolute changes in U are largest are extracted and used to 
    craft simple explainations for the user. The feature names are created i feature_interpretations.py
    Returns raw explanation data.
    """

    # Extracting the movie indices 
    movie_index = list(R.columns).index(movie_to_rate_id)
    movie_vector = V[movie_index]
    all_explanations = []

    # Constrcuting the baseline vector
    cold_start = ColdStart(V, R)
    baseline_user_vector = cold_start.update_user_vector(current_user_ratings)

    for hypothetic_rating in range(1, 6):
        # Simulating user ratings
        simulated_ratings = current_user_ratings.copy()
        simulated_ratings[movie_to_rate_id] = hypothetic_rating

        # Recomputing U and the feature deltas
        updated_user_vector = cold_start.update_user_vector(simulated_ratings)
        raw_feature_deltas = updated_user_vector - baseline_user_vector
    
        # Predicting scores for all movies with the updated user vector 
        predicted_ratings = np.clip(V @ updated_user_vector, 0, 5)

        # We only consider movies that the user has not rated yet
        rated_ids = set(simulated_ratings.keys())
        unrated_indices = [idx for idx, movie_id in enumerate(R.columns) if movie_id not in rated_ids]

        top_index = max(unrated_indices, key=lambda i: predicted_ratings[i])
        top_movie_id = R.columns[top_index]
        top_movie_title = movieId_to_title.get(top_movie_id, "Unknown")
        top_movie_vector = V[top_index]

        # Ranking features by absolute change in magnitude and selecting top k
        top_feature_indices = np.argsort(np.abs(raw_feature_deltas))[::-1][:3]

        feature_changes = []
        for idx in top_feature_indices:
            # Direction of user feature change
            direction = "increased" if raw_feature_deltas[idx] > 0 else "decreased"
            movie_val = top_movie_vector[idx]
            user_val = updated_user_vector[idx]
            match = "aligns well" if np.sign(raw_feature_deltas[idx]) == np.sign(movie_val) else "differs"
            # Human readanle label + description
            feature_title = feature_dict['Feature_' + str(idx + 1)]
            feature_info = feature_characteristics['Feature_' + str(idx + 1)]
            feature_changes.append((feature_title,
                                    direction,
                                    feature_info,
                                    user_val,
                                    movie_val,
                                    match,
                                    raw_feature_deltas[idx]))

        # Accumulate raw data 
        explanation_data = (
            hypothetic_rating,
            top_movie_id,
            top_movie_title,
            predicted_ratings[top_index],
            feature_changes
        )
        all_explanations.append(explanation_data)

        # Still print for console output (keep existing behavior)
        print(f"→ If you rate it a {hypothetic_rating}:")
        print(f"  Next recommended movie: '{top_movie_title}' (ID: {top_movie_id})"
              f"with predicted rating {predicted_ratings[top_index]:.2f}")
        print("  Why:")
        for feature_title, direction, feature_info, user_val, movie_val, match, delta_val in feature_changes:
            print(
                f"   - The feature '{feature_title}' {direction} in your profile, and this movie {match} with that change (score: {movie_val:.2f})")
            # print(f"   - Feature '{feature_title}' {direction} by {delta_val:.3f}, "
            #     f"user={user_val:.2f}, movie={movie_val:.2f} ({match})")
        print()

    return all_explanations


# ----------- Active learning -----------


def active_learning_step(initial_user_vector, V, R, user_ratings, movieId_to_title=None, skipped_movies=None):
    """
    A single selection step of the active learning loop. It takes the current user vector U, 
    predict score for all movies, filters out already-rated ones and returns the argmax. 
    Returns: top_movie_id, top_movie_title, predicted_rating
    """

    if movieId_to_title is None:
        movieId_to_title = load_movie_data()

    predicted_ratings = np.clip(V @ initial_user_vector, 0, 5)
    R.columns = R.columns.astype(int)

    # Find indices of movies not yet rated
    unrated_indices = [idx for idx, movie_id in enumerate(R.columns) if movie_id not in user_ratings.keys()]
    # If skipped_movies is provided, filter out those movies from unrated_indices
    not_skipped_indices = [idx for idx in unrated_indices if
                           R.columns[idx] not in skipped_movies] if skipped_movies else unrated_indices

    top_index = max(not_skipped_indices, key=lambda i: predicted_ratings[i])
    top_movie_id = R.columns[top_index]
    top_movie_title = movieId_to_title.get(top_movie_id, "Unknown title")

    print(
        f"Recommended movie: '{top_movie_title}' (movieID: {top_movie_id}) with predicted rating: {predicted_ratings[top_index]:.3f}")

    # Return raw values
    return top_movie_id, top_movie_title, predicted_ratings[top_index]


def active_learning_loop(initial_user_vector, V, R, user_ratings, max_rounds=3, movieId_to_title=None):
    """
    The user repeatedly rates selected movies and the user latent vector is updated accordingly
    Returns: The updated user latent vector
    """

    if movieId_to_title is None:
        movieId_to_title = load_movie_data()

    U = initial_user_vector
    rated_ids = set(user_ratings.keys())

    for i in range(max_rounds):
        top_movie_id, top_movie_title, _ = active_learning_step(U, V, R, user_ratings, movieId_to_title)
        print(f"Round {i+1}: Recommended '{top_movie_title}' (ID: {top_movie_id})")

        explain_impact(user_ratings, top_movie_id, V, R, movieId_to_title, top_k=3)
        rating = simulate_rating()
        user_ratings[top_movie_id] = rating
        rated_ids.add(top_movie_id)
        print(f"Simulated user rating: {rating}\n")
        cold_start = ColdStart(V, R)
        U = cold_start.update_user_vector(user_ratings)

    return U


# --- Demo: cold-start simulation --- #

def get_initial_ratings():
    """
    Obtaining randomly simulated initial ratings on the initially selected movies (one from each cluster to ensure diversity and informativity).
    Returns a dictionary mapping the movie ID's to their ratings
    """

    initial_movies = get_initial_movies()

    user_rating = {}
    for movie in initial_movies:
        random_rating = random.choice([1, 2, 3, 4, 5])
        user_rating[movie['movieId']] = random_rating
        print("Movie title:", movie['title'], "Movie_ID:", movie['movieId'])
    return user_rating


def simulate_rating():
    return random.choice([1, 2, 3, 4, 5])


def run_cold_start_demo():
    # Loading the rating matrix R, user matrix U, movie matrix V
    model, R, U, V = get_R_U_V()

    # Simulating initial rating on the cold start movies
    user_ratings = get_initial_ratings()

    # Initializing the cold start recommender and estimating the initial user vector
    cold_start_recommender = ColdStart(V, R)
    user_vector = cold_start_recommender.update_user_vector(user_ratings)

    # Refining the user vector through more ratings
    final_user_vector = active_learning_loop(user_vector, V, R, user_ratings, max_rounds=3)

    # Displaying the final user vector with human-readable feature names (found in feature_interpretations.py)
    for i, value in enumerate(final_user_vector):
        print(f" Feature {feature_dict['Feature_' + str(i + 1)]}: {value:2f}")

    print(f"Final user vector:", final_user_vector)

if __name__=="__main__":
    run_cold_start_demo()
