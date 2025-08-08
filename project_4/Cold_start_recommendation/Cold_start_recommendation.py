from Clustering import get_selected_cold_start_movies
from Factorization_engine import get_R_U_V 
from feature_interpretations import feature_dict, feature_characteristics
import pandas as pd
import os
import numpy as np
import random
from numpy.linalg import norm

# Importing df_movies to create a conversion dictironay between movie ID and title
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path_movies = os.path.join(BASE_DIR, 'data', 'ml_latest_small', 'movies.csv')
df_movies = pd.read_csv(csv_path_movies)
movieId_to_title = dict(zip(df_movies['movieId'], df_movies['title']))


# ---------- Cold start ---------- #
# Learning the initial preference vector U_i based on the 3 initial rated movies 

# TODO: Integrate this into the interface to get the actual ratings of the user. 

# Obtaining the initial ratings on 3 randomly selected movies (one from each cluster to ensure diversity and informativity).
def get_initial_ratings():
    selected_movies = get_selected_cold_start_movies() 
    initial_movies = random.sample(selected_movies, 3) 
    print(f"Initial movies:", initial_movies)

    # Only temporary. Just for testing:
    user_rating = {}
    for movie in initial_movies:
        random_rating = random.choice([1,2,3,4,5])
        user_rating[movie['movieId']] = random_rating 
        print("Movie title:", movie['title'], "Movie_ID:", movie['movieId'])
    return user_rating 


# Learning the initial user vector (after the first 3 movies have been rated)
class ColdStart:
    def __init__(self, V_matrix, R_matrix, lambd=0.1):
        self.V = V_matrix # Latent ratings matrix
        self.K = V_matrix.shape[1]
        self.R = R_matrix # Original matrix containing each rating for each user
        self.R.columns = self.R.columns.astype(int)
        self.lambd = lambd
     
    def update_user_vector(self, user_ratings):
        # Extracting the rated movies and ensuring they are found in the R-matrix (i.e. have some rating from before)
        rated_ids = list(user_ratings.keys())
        found_ids = [i for i in rated_ids if i in self.R.columns]
        if found_ids != rated_ids:
            raise ValueError("All rated ids were not found. The found ones were:", found_ids)
        
        ratings = np.array(list(user_ratings.values()), dtype=np.float64)
        rated_indices = [self.R.columns.get_loc(i) for i in rated_ids if i in self.R.columns]
        V_rated = self.V[rated_indices,:]

        # Learning a new user vector U_i (using eq.2 from the instruction pdf)
        A = V_rated.T @ V_rated + self.lambd * np.eye(self.K)
        b = V_rated.T @ ratings
        U_i = np.linalg.solve(A,b)
        return U_i
    

def cosine_similarity(a,b):
    return np.dot(a,b) / (norm(a) * norm(b)) if norm(a) > 0 and norm(b) > 0 else 0

def explain_impact(current_user_ratings, movie_to_rate_id, V, R, movieId_to_title, top_k=1, feature_dict = feature_dict, feature_information = feature_characteristics):
    movie_index = list(R.columns).index(movie_to_rate_id)
    movie_vector = V[movie_index]

    for hypothetic_rating in range(1,6):
        simulated_ratings = current_user_ratings.copy()
        simulated_ratings[movie_to_rate_id] = hypothetic_rating

        cold_start = ColdStart(V,R)
        updated_user_vector = cold_start.update_user_vector(simulated_ratings)
        predicted_ratings = V @ updated_user_vector
        predicted_ratings = np.clip(predicted_ratings, 0, 5)

        rated_ids = set(simulated_ratings.keys())
        unrated_indices = [idx for idx, movie_id in enumerate(R.columns) if movie_id not in rated_ids]

        top_index = max(unrated_indices, key=lambda i: predicted_ratings[i])
        top_movie_id = R.columns[top_index]
        top_movie_title = movieId_to_title.get(top_movie_id, "Unknown")
        top_movie_vector = V[top_index]

        feature_deltas = updated_user_vector - ColdStart(V,R).update_user_vector(current_user_ratings)
        top_feature_indices = np.argsort(np.abs(feature_deltas))[::-1][:3]
        feature_explanations = []

        for idx in top_feature_indices:
            direction = "increased" if feature_deltas[idx] > 0 else "decreased"
            user_val = updated_user_vector[idx]
            movie_val = top_movie_vector[idx]
            match = "aligns well" if np.sign(feature_deltas[idx]) == np.sign(movie_val) else "differs"

            feature_title = feature_dict['Feature_' + str(idx+1)]
            feature_info = feature_characteristics['Feature_' + str(idx+1)]

            feature_explanations.append(
                f"The feature '{feature_title}' {direction} in your profile, and this movie {match} with that change (score: {movie_val:.2f})"
            )

        print(f"→ If you rate it a {hypothetic_rating}:")
        print(f"  Next recommended movie: '{top_movie_title}' (ID: {top_movie_id})")
        print("  Why:")
        for explanation in feature_explanations:
            print(f"   - {explanation}")
        print()

def active_learning_loop(initial_user_vector, V, R, user_ratings, max_rounds=3):
    U = initial_user_vector
    rated_ids = set(user_ratings.keys())
    for i in range(max_rounds):
        predicted_ratings = V @ U # Predicted ratings for all movies 
        
        predicted_ratings = np.clip(predicted_ratings, 0, 5)

        R.columns = R.columns.astype(int)
        unrated_indices = [idx for idx, movie_id in enumerate(R.columns) if movie_id not in rated_ids] #movies not yet rated by the user

       # Selecting the movie with the highest predicted rating 
        top_index = max(unrated_indices, key=lambda i:predicted_ratings[i]) 
        top_movie_id = R.columns[top_index] 
        top_movie_title = movieId_to_title.get(top_movie_id, "Unknown title")
        
        print(f" Round {i+1}: Recommended movie: '{top_movie_title}' (movieID: {top_movie_id}) with predicted rating: {predicted_ratings[top_index]:.3f}")
        print(f"Let's see how your ratings will affect your next recommendation:")
        explain_impact(user_ratings, top_movie_id, V, R, movieId_to_title)
        simulated_rating = random.choice([1,2,3,4,5])
        user_ratings[top_movie_id] = simulated_rating
        rated_ids.add(top_movie_id)
        cold_start = ColdStart(V,R)
        U = cold_start.update_user_vector(user_ratings)

    return U


model, R, U, V = get_R_U_V()

user_ratings = get_initial_ratings()

cold_start_recommender = ColdStart(V, R)
user_vector = cold_start_recommender.update_user_vector(user_ratings)

final_user_vector = active_learning_loop(user_vector, V, R, user_ratings, max_rounds=3)
for i, value in enumerate(final_user_vector):
    print(f" Feature {feature_dict['Feature_' + str(i+1)]}: {value:2f}")

print(f"Final user vector:", final_user_vector)
