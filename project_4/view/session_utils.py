import logging
import os
import pandas as pd
from django.conf import settings

from project_4.Cold_start_recommendation.Cold_start import (
    ColdStart,
    get_R_U_V,
    load_movie_data,
    active_learning_step
)

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
INITIAL_MOVIES_COUNT = 5
MAX_ACTIVE_LEARNING_ROUNDS = 4


class MovieCache:
    """Separate cache instance for each view"""

    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.R_matrix = None
        self.V_matrix = None
        self.movieId_to_title = None
        self.loaded = False
        self.selected_movies = None
        self.sampled_movies = None

    def ensure_loaded(self):
        """Load movie dataset and matrix factorization results"""
        if self.loaded:
            return

        try:
            movies_path = os.path.join(settings.BASE_DIR, 'data', 'ml_latest_small', 'movies.csv')
            ratings_path = os.path.join(settings.BASE_DIR, 'data', 'ml_latest_small', 'ratings.csv')

            self.movies_df = pd.read_csv(movies_path)
            self.ratings_df = pd.read_csv(ratings_path)
            model, self.R_matrix, U_matrix, self.V_matrix = get_R_U_V()
            self.movieId_to_title = load_movie_data()

            self.loaded = True
            logger.info("Movie data loaded successfully")

        except Exception as e:
            logger.error(f"Error loading movie data: {str(e)}")
            raise

    def get_initial_movie_selection(self, n):
        """Get initial movies using clustering"""
        if self.selected_movies is None:
            from project_4.Cold_start_recommendation.Clustering import run_true_hybrid_cold_start

            class MockModel:
                def __init__(self, V):
                    self.V = V

            mock_model = MockModel(self.V_matrix)

            selected_movies, _, _, _ = run_true_hybrid_cold_start(
                df_movies=self.movies_df,
                df_ratings=self.ratings_df,
                mat_fac_model=mock_model,
                R_matrix=self.R_matrix,
                n_clusters=n
            )

            self.selected_movies = selected_movies

        return self.selected_movies


class ColdStartSession:
    """Session state management for cold start recommendation"""

    def __init__(self, session_id=None):
        self.session_id = session_id or f'session_{id(self)}'
        self.reset()

    def reset(self):
        """Reset session to initial state"""
        self.is_initialized = False
        self.current_step = 0
        self.total_ratings = 0
        self.user_vector = None
        self.current_movies = []
        self.current_movie_index = 0
        self.rated_movies = []
        self.skipped_movies = []
        self.initial_movies_count = INITIAL_MOVIES_COUNT
        self.active_learning_round = 0
        self.max_active_learning_rounds = MAX_ACTIVE_LEARNING_ROUNDS

    def store_rating(self, movie_id, rating):
        """Store a rating and increment counters"""
        rating_data = {
            'movie_id': movie_id,
            'rating': rating,
            'step': self.current_step
        }
        self.rated_movies.append(rating_data)
        self.total_ratings += 1
        self.current_step += 1
        logger.info(f"Stored rating: {movie_id} -> {rating}")

    def store_skip(self, movie_id):
        """Store a skip"""
        skip_data = {
            'movie_id': movie_id,
            'step': self.current_step
        }
        self.skipped_movies.append(skip_data)

    def get_rated_dict(self):
        """Get ratings as dictionary"""
        return {d['movie_id']: d['rating'] for d in self.rated_movies}

    def get_skipped_ids(self):
        """Get list of skipped movie IDs"""
        return [d['movie_id'] for d in self.skipped_movies]

    def update_user_vector(self, movie_cache):
        """Update user vector with all current ratings"""
        rated_dict = self.get_rated_dict()
        cold_start = ColdStart(movie_cache.V_matrix, movie_cache.R_matrix)
        self.user_vector = cold_start.update_user_vector(rated_dict)
        logger.info(f"Updated user vector. Total ratings: {len(rated_dict)}")

    def get_session_context(self):
        """Get current session state for responses"""
        if not self.is_initialized:
            return {}

        return {
            'session_active': True,
            'current_step': self.current_step,
            'initial_movies_count': self.initial_movies_count,
            'total_ratings': self.total_ratings,
            'rated_count': len(self.rated_movies),
            'skipped_count': len(self.skipped_movies),
            'current_movies': self.current_movies,
            'session_phase': 'initial_rating' if self.current_step < self.initial_movies_count else 'active_learning',
            'active_learning_round': self.active_learning_round,
            'max_active_learning_rounds': self.max_active_learning_rounds
        }

    def is_in_initial_phase(self):
        """Check if still in initial rating phase"""
        return self.current_step < self.initial_movies_count

    def is_session_complete(self):
        """Check if session is complete"""
        return self.active_learning_round > self.max_active_learning_rounds


def initialize_session(session, movie_cache, initial_count=None):
    """Initialize cold start session"""
    if initial_count is None:
        initial_count = INITIAL_MOVIES_COUNT

    movie_cache.ensure_loaded()
    raw_movies = movie_cache.get_initial_movie_selection(initial_count)

    initial_movies = [
        {
            'movie_id': m['movieId'],
            'title': m['title'],
            'genres': m['genres'],
        }
        for m in raw_movies
    ]

    session.is_initialized = True
    session.current_movies = initial_movies
    session.initial_movies_count = initial_count

    return initial_movies


def get_next_movie_in_phase(session, movie_cache):
    """Get next movie based on current phase"""
    if session.is_in_initial_phase():
        return get_next_initial_movie(session)
    else:
        return get_next_active_learning_movie(session, movie_cache)


def get_next_initial_movie(session):
    """Get next movie in initial rating phase"""
    initial_movies = session.current_movies
    current_index = session.current_movie_index + 1
    session.current_movie_index = current_index

    if current_index < len(initial_movies):
        return initial_movies[current_index]
    else:
        return None


def get_next_active_learning_movie(session, movie_cache):
    """Get next movie in active learning phase using active learning step"""
    if session.is_session_complete():
        return None

    try:
        # Use the existing active_learning_step function
        top_movie_id, top_movie_title, predicted_rating = active_learning_step(
            session.user_vector,
            movie_cache.V_matrix,
            movie_cache.R_matrix,
            session.get_rated_dict(),
            movie_cache.movieId_to_title,
            session.get_skipped_ids()
        )

        # Get additional movie info (genres) from movies dataframe
        movie_info = movie_cache.movies_df[
            movie_cache.movies_df['movieId'] == top_movie_id
            ].iloc[0]

        # Transform to what GUI expects
        movie_data = {
            'movie_id': int(top_movie_id),
            'title': top_movie_title,
            'genres': movie_info['genres'],
            'predicted_rating': float(predicted_rating)
        }

        logger.info(f"Active learning selected: {top_movie_title} "
                    f"(predicted: {predicted_rating:.2f})")

        return movie_data

    except Exception as e:
        logger.error(f"Error in active learning step: {str(e)}")
        return None


def process_rating_submission(session, movie_cache, movie_id, rating):
    """Process rating submission and determine next action"""
    session.store_rating(movie_id, rating)
    session.update_user_vector(movie_cache)

    # Determine next movie and messages
    if session.is_in_initial_phase():
        next_movie = get_next_initial_movie(session)
        remaining = session.initial_movies_count - session.current_step
        message = f"Rating submitted! {remaining} more movies to rate."
        session_step = 'initial_rating'
    else:
        # Transitioning to or continuing active learning
        if session.current_step == session.initial_movies_count:
            message = "Initial rating complete! Moving to personalized recommendations."
            session.active_learning_round = 1
        else:
            session.active_learning_round += 1
            round_num = session.active_learning_round
            max_rounds = session.max_active_learning_rounds
            message = f"Rating submitted! Round {round_num}/{max_rounds} of active learning."

        session_step = 'active_learning'

        if session.active_learning_round <= session.max_active_learning_rounds:
            next_movie = get_next_active_learning_movie(session, movie_cache)
        else:
            message = "Active learning session complete! Thank you for your participation."
            next_movie = None

    return next_movie, message, session_step


def process_movie_skip(session, movie_cache, movie_id):
    """Process movie skip and determine next action"""
    session.store_skip(movie_id)

    if session.is_in_initial_phase():
        next_movie = get_next_initial_movie(session)
        message = "Movie skipped. Continue with initial rating."
        session_step = 'initial_rating'

        if next_movie is None:
            message = "Initial movies completed."
    else:
        # Active learning phase - get next recommendation
        next_movie = get_next_active_learning_movie(session, movie_cache)
        round_num = session.active_learning_round
        max_rounds = session.max_active_learning_rounds
        message = f"Movie skipped. Round {round_num}/{max_rounds} of active learning."
        session_step = 'active_learning'

        if next_movie is None:
            message = "Active learning session complete! Thank you for your participation."

    return next_movie, message, session_step


def generate_rating_explanation(session, movie_cache, movie_id):
    """Generate explanations for different rating impacts"""
    from project_4.Cold_start_recommendation.Cold_start import explain_impact

    current_ratings = session.get_rated_dict()

    # Use the existing explain_impact function
    raw_explanations = explain_impact(
        current_ratings,
        movie_id,
        movie_cache.V_matrix,
        movie_cache.R_matrix,
        movie_cache.movieId_to_title
    )

    # Transform raw data to GUI format
    formatted_explanations = []
    for rating, next_movie_id, next_movie_title, next_predicted_rating, feature_changes in raw_explanations:

        # Transform feature changes - now including delta values
        transformed_features = []
        for feature_title, direction, feature_info, user_val, movie_val, match, delta_val in feature_changes:
            transformed_features.append({
                'feature_name': feature_title,
                'change': 'increase' if direction == 'increased' else 'decrease',
                'description': f"{direction.capitalize()} your preference for {feature_info.lower()}",
                'delta_value': float(delta_val),
                'user_value': float(user_val),
                'movie_value': float(movie_val),
                'match': match
            })

        # Calculate confidence based on rating extremeness
        confidence = min(0.95, max(0.4, 0.5 + abs(rating - 3) * 0.2))

        formatted_exp = {
            'rating': rating,
            'explanation': {
                'predicted_next_movie': {
                    'title': next_movie_title,
                    'movie_id': int(next_movie_id),
                    'confidence': confidence
                },
                'feature_changes': transformed_features,
            }
        }
        formatted_explanations.append(formatted_exp)

    return formatted_explanations