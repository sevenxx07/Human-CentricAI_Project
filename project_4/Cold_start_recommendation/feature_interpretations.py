import os
from collections import Counter

import numpy as np
import pandas as pd

from pbl.settings import BASE_DIR
from .Factorization_engine import get_R_U_V

csv_path_movies = os.path.join(BASE_DIR, 'data', 'ml_latest_small', 'movies.csv')

feature_dict = {
 "Feature_1" : "Offbeat Capers & Musical Mischief",
 "Feature_2": "Sci-Fi Romance with a Witty Edge",
 "Feature_3": "Whimsical Romances & Daydream Escapes",
 "Feature_4" : "Passion & Peril in Tumultuous Times",
 "Feature_5": "Twisted Humor & Creepy Vibes",
 "Feature_6": "Poetic Quests & Fantastical Journeys",
 "Feature_7" : "Cosmic Comedy & Social Satire",
 "Feature_8": " Intense Romances & Rebellious Thrills",
 "Feature_9": "Low-Key Action & Quiet Intrigue",
 "Feature_10" : "Classic Suspense with a Romantic Core",
 "Feature_11": "Whimsical Worlds for All Ages"
 }

feature_characteristics = {
 "Feature_1" : "Quirky Ccrime, satirical comedy, and occasional musical numbers",
 "Feature_2": "Lighthearted or emotional stories mixing love, tech, and humor",
 "Feature_3": "Dreamu love stories, gentre thrillers, and magical realism",
 "Feature_4" : "Love, war, and danger collide in high-stakes drama",
 "Feature_5": "Where horror meets irony -- funny, freaku, and off-kilter",
 "Feature_6": "Emotionally rich adventures with surreal or mysitcal tones",
 "Feature_7" : "Sci-fi and satire with a splash of existential humor",
 "Feature_8": "Stylized love stories, edgy drama, and emotional heat",
 "Feature_9": "Tense relationships and subtle thrillers with emotional depth",
 "Feature_10" : "Mysteries and thrillers grounded in emotional stakes",
 "Feature_11": "Lighthearted, imaginative stories for families and dreamers."
 }


# Contextualization of the features.
# Purpose: To provide the user with actual, interpretable information about how their choices will affect
# the next movie to be presented.
def interpret_latent_features(V, df_movies, movieId_to_title, top_n=20):

    feature_interpretations = []

    movie_ids = list(movieId_to_title.keys())
    movie_index_to_ids = {i: movie_ids[i] for i in range(len(movie_ids))}
    df_movies = df_movies.set_index('movieId')
    num_features = V.shape[1]

    for feature_index in range(num_features):
        top_indices = np.argsort(V[:, feature_index])[::-1][:top_n]
        top_movie_ids = [movie_index_to_ids[i] for i in top_indices]
        top_titles = [movieId_to_title[i] for i in top_movie_ids]

        top_genres = []
        for i in top_movie_ids:
            if i in df_movies.index:
                genres = df_movies.loc[i, "genres"]
                if isinstance(genres, str):
                    top_genres.extend(genres.split('|'))

        genre_couts = Counter(top_genres)

        summary = {
            'Feature': feature_index +1,
            'Top Genres': genre_couts.most_common(5),
            'Top Movies': top_titles[:5]
        }

        feature_interpretations.append(summary)

    return feature_interpretations


def __main__():
    # Load movies data if not provided
    df_movies = pd.read_csv(csv_path_movies)
    movieId_to_title = dict(zip(df_movies['movieId'], df_movies['title']))

    # Load the R matrix and perform matrix factorization
    model, R, U, V = get_R_U_V()

    # Interpret the latent features
    feature_information = interpret_latent_features(V, df_movies, movieId_to_title)

    # Print the feature interpretations
    for feature in feature_information:
        print(f"Feature {feature['Feature']}")
        print("Top Genres:", feature['Top Genres'])
        print("Example Movies:", feature['Top Movies'])
        print()



# Feature 1: Drama (10), Comedy (6), Crime (3), Musical (3), Horror (3)
# "Quirky Crime & Musical Escapades"
# Vibe: Stylized, offbeat stories with a blend of humor, music and crime.
# Example Movies: Dr.Horrible's Sing-Along Blog, Merry Madagascar, Three Colors: Red


# Feature 2: Drama (10), Comedy(8), Sci-Fi (5), Romance (4), Horrror (3)
# "Sci-Fi Romance with a comedic twist"
# Vibe: Romantic and humourous films, often with fantastical or science fiction elements
# Example Movies: Grabbers, Gone with the Wind, Shakespear in Love


# Feature 3: Drama (10), Comedy (8), Romance (7), Thriller (3), Fantasy (3)
# "Dreamy Rom-Coms with a Touch of Magic"
# Vibe: Romantic fantasies and emotionally risch stories with surreal or magical overtones
# Example Movies: Charade, Fantastia, Beature of the Day


# Feature 4: Drama (12), Romance (5), Thriller (5), War (3), Thriller (4)
# "Love in Times of War & Tension"
# Vibe: High-stakes romantic and war-time dramas with intense emotional or politcal themes
# Example movie: From here to eterity, Universal Soldier, Sleepy Hollow


# Feature 5: Drama (12), Comedy (8), Horror (5), Romance (4), Thriller (4)
# "Dark Comedy Meets Horror"
# Vibe: Horror and thriller elements combines with humor and romance -- often edgy or ironic
# Example Movies: Creepshow 2, Arachnophobia, Curly Sue


# Feature 6: Drama (10), Comedy (5), Adventure (5), Fantasy (4), Action (4)
# "Fantastical Adventures with Heart"
# Vibe: Surreal, emotionally-charged adventures, often poetic or philosophical
# Example Movies: Dreams, In the Name of the Father, My Name is Bruce


# Feature 7: Drama (12), Comedy (9), Adventure (4), Crime (4), Sci-Fi (3)
# "Sci-Fi & Social Commentary with Laughs"
# Vibe: Introspective or socially aware films that mix science fiction with humor and satire"
# Example movies: 2001: A Space Adyssey, Big Chill, 99 Francs


# Feature 8: Drama (14), Comedy (6), Romance (5), Thriller (3), Action (3)
# "Edge Romance & Stylized Thrillers"
# Vibe: Intimate relationships, rebellious characters, and dramatic twists
# Example Movies: Before Sunrise, Clockwork Orange, His Girl Friday


# Feature 9: Drama (11), Thriller (4), Comedy (4), Romance (4), Action (2)
# "Mystery & Relaationships on the edge"
# Vibe: Character-driven stories that blend romance or huor with suspense and danger
# Example Movie: Raiders of the Lost Ark, Trust, Seven Days in May


# Feature 10: Drama (11), Thriller (6), Romance (5), Mystery (5), Comedy (4)
# "Emotional Thrillers & Classic Myseteries"
# Vibe: Tense and insprospective stories witgh love, secrets, and string plot twists.
# Example Movies: Bound, Out of the Past, Jean de Florette.


# Feature 11: Drama (8), Comedy (7), Fantasy (5), adventure (4), Children (4)
# "Whimsical Family-Friendly Fantasies"
# Vibe: Light-hearted, often fantastical or satirical films that span generations
# Example Movies: Wallace and Gromit, Dr.No, Borat, Shakespeare in Love.