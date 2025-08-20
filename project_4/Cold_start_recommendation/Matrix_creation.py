import os
import sys
import django
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbl.settings')
django.setup()

from django.conf import settings

def create_rating_matrix():
    """
    Create the rating matrix from MovieLens data and save it to the data directory
    """

    csv_path_movies = os.path.join(settings.BASE_DIR, 'data', 'ml_latest_small', 'movies.csv')
    # Format: movieId, title, genres
    csv_path_ratings = os.path.join(settings.BASE_DIR, 'data', 'ml_latest_small', 'ratings.csv')
    # Format: userId, movieId, rating, timestamp
    output_path = os.path.join(settings.BASE_DIR, 'data', 'R_matrix.csv')

    print(f"Reading data")
    
    df_movies = pd.read_csv(csv_path_movies)
    df_ratings = pd.read_csv(csv_path_ratings)
    #TODO so which normalisation to use
    #df_ratings['rating']=(df_ratings['rating']-df_ratings['rating'].mean())/df_ratings['rating'].std()
    df_ratings['rating']=(df_ratings['rating']-df_ratings['rating'].min())/(df_ratings['rating'].max() - df_ratings['rating'].min())
    
    print(df_movies.head())
    print(df_ratings[:20])

    userId = df_ratings.iloc[:,0]
    R = df_ratings.pivot(index = 'userId', columns = 'movieId', values = 'rating')
    R = R.replace(np.nan, None)

    R.to_csv(output_path)
