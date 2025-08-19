import os
import sys
import django
import pandas as pd
import numpy as np

sys.path.append('/Users/stinahellgren/Documents/Human AI/Human-CentricAI_Project')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbl.settings')
django.setup()

from django.conf import settings

csv_path_movies = os.path.join(settings.BASE_DIR, 'data', 'ml_latest_small', 'movies.csv')
# Format: movieId, title, genres 
csv_path_ratings = os.path.join(settings.BASE_DIR, 'data', 'ml_latest_small', 'ratings.csv')
# Format: userId, movieId, rating, timestamp 

print(f"reading data")

df_movies = pd.read_csv(csv_path_movies)
df_ratings = pd.read_csv(csv_path_ratings)
#df_ratings['rating']=(df_ratings['rating']-df_ratings['rating'].mean())/df_ratings['rating'].std()
df_ratings['rating']=(df_ratings['rating']-df_ratings['rating'].min())/(df_ratings['rating'].max() - df_ratings['rating'].min())

print(df_movies.head())
print(df_ratings[:20])

userId = df_ratings.iloc[:,0]
R = df_ratings.pivot(index = 'userId', columns = 'movieId', values = 'rating')
R = R.replace(np.nan, None)
R.to_csv("R_matrix.csv")
