from Clustering import run_true_hybrid_cold_start
from Clustering import get_selected_cold_start_movies
from Factorization_engine import Matrix_Factorization
from Factorization_engine import get_R_U_V 
import pandas as pd
import os
import json
import numpy as np
import random

# ----------1. Cold start ----------#
# Learning the initial preference vector U_i based on the 3 initial rated movies 

# TODO: Integrate this into the interface to get the actual ratings of the user. 
def get_initial_ratings():
    selected_movies = get_selected_cold_start_movies()
    initial_movies = random.sample(selected_movies,3)
    user_rating = {}
    for i in range(3):
        random_rating = random.choice([1,2,3,4,5])
        user_rating[initial_movies[i]['movieId']] = random_rating 
    return user_rating 

class ColdStart:
    def __init__(self, V_matrix, R_matrix, movie_to_index, lambd=0.1):
        self.V = V_matrix
        self.K = V_matrix.shape[1]
        self.R = R_matrix
        self.movie_to_index = movie_to_index
        self.lambd = lambd

    def update_user_vector(self, user_ratings):
        rated_ids = list(user_ratings.keys())
        ratings = np.array(list(user_ratings.values()), dtype=np.float64)

        indices = [self.movie_to_index[ids] for ids in rated_ids if ids in self.movie_to_index]
        if not indices:
            print("No matching movie IDs found in `movie_to_index`.")
            print("Rated movie IDs:", rated_ids)
            print("Available movie IDs:", list(self.movie_to_index.keys())[:10])
            return np.zeros(self.K)
        
        V_rated = self.V[indices,:]

        A = V_rated.T @ V_rated + self.lambd * np.eye(self.K)
        b = V_rated.T @ ratings
        U_i = np.linalg.solve(A,b)
        return U_i

# ---------- 2. Active learning -------------
# a) Predicting ratings for all unrated movies using the current U
# b) Select the next movie to display to the user
# c) Update the user vector after they've rated the next movie
# d) Explaining how the user's choice will affect the next movie

def active_learning_loop(initial_user_vector, V, movie_to_index, index_to_movie, user_ratings, max_rounds = 3):
    U = initial_user_vector
    rated_ids = set(user_ratings.keys())

    for i in range(max_rounds):
        predicted_ratings = V @ U
        
        unrated_indices = [i for i in range(V.shape[0]) if index_to_movie[i] not in rated_ids]
        top_index = max(unrated_indices, key=lambda i:predicted_ratings[i])
        top_movie_id = index_to_movie[top_index]
        #top_movie_title = get_movie_title(top_movie_id)
pass 


# ---------- 3. Recommendation  -------------
# a) Recompute the final user vector
# b) Recommend the highest predicted rating not yet rated 

model, R, U, V = get_R_U_V()

movie_ids = [int(col) for col in R.columns]
movie_to_index = {ids: index for index, ids in enumerate(movie_ids)}

user_ratings = get_initial_ratings()
recommender = ColdStart(V, R, movie_to_index)
user_vector = recommender.update_user_vector(user_ratings)
print(user_vector)
