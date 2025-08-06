from Clustering import run_true_hybrid_cold_start
from Clustering import get_selected_cold_start_movies
from Factorization_engine import Matrix_Factorization
from Factorization_engine import get_R_U_V 
import pandas as pd
import os
import json
import numpy as np
import random

# ---------- Cold start ---------- #
# Learning the initial preference vector U_i based on the 3 initial rated movies 

# TODO: Integrate this into the interface to get the actual ratings of the user. 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

csv_path_movies = os.path.join(BASE_DIR, 'data', 'ml_latest_small', 'movies.csv')
csv_path_ratings = os.path.join(BASE_DIR, 'data', 'ml_latest_small', 'ratings.csv')
df_movies = pd.read_csv(csv_path_movies)
df_ratings = pd.read_csv(csv_path_ratings)
movieId_to_title = dict(zip(df_movies['movieId'], df_movies['title']))



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

#print("Initial rating:", get_initial_ratings())

class ColdStart:
    def __init__(self, V_matrix, R_matrix, lambd=0.1):
        self.V = V_matrix
        self.K = V_matrix.shape[1]
        self.R = R_matrix
        self.R.columns = self.R.columns.astype(int)
        self.lambd = lambd
     
    def update_user_vector(self, user_ratings):
        rated_ids = list(user_ratings.keys())
        
        print("Rated ids:", rated_ids)
        
        found_ids = [i for i in rated_ids if i in self.R.columns]
        ratings = np.array(list(user_ratings.values()), dtype=np.float64)
        
        print("Found ids:", found_ids)

        if found_ids != rated_ids:
            raise ValueError("All rated ids were not found. The found ones were:", found_ids)

        rated_indices = [self.R.columns.get_loc(i) for i in rated_ids if i in self.R.columns]

        V_rated = self.V[rated_indices,:]

        # if len(rated_indices) == 0:
        #     raise ValueError("None of the rated movie IDS were found")
        # Learning a new user vector U_i (eq.2 from instructions)
        A = V_rated.T @ V_rated + self.lambd * np.eye(self.K)
        b = V_rated.T @ ratings
        U_i = np.linalg.solve(A,b)
        return U_i

# In the active learning loop, a prediction is made forn unrated ,ovies and the 
# top predicted movie is selected. The user is asked to rate it, and the vector is 
# once again updated. 
def active_learning_loop(initial_user_vector, V, R, user_ratings,  max_rounds = 3):
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

        # TODO: Replace this by the real interface-suitable code. 
        simulated_rating = random.choice([1,2,3,4,5])
        user_ratings[top_movie_id] = simulated_rating
        
        rated_ids.add(top_movie_id) # Marking the movie as rated 

        cold_start = ColdStart(V,R) # Update the user vector 
        U = cold_start.update_user_vector(user_ratings)

    
    return U


def explain_impact(current_user_ratings, movie_to_rate_id, V, R, movieId_to_title, top_k=1):

    explainations = {}
    for rating in range(1,6):
        pass


model, R, U, V = get_R_U_V()
#print("Movie IDS:",movie_ids)
# print("R shape:", R.shape)
# print("V shape:", V.shape)
# print("U shape:", U.shape)

# print(type(R))               
# print(R.columns[:20])       
# print(R.columns.dtype) 



user_ratings = get_initial_ratings()
cold_start_recommender = ColdStart(V, R)
user_vector = cold_start_recommender.update_user_vector(user_ratings)
final_user_vector = active_learning_loop(user_vector, V, R, user_ratings, max_rounds=3)

print(f"Final user vector:", final_user_vector)
