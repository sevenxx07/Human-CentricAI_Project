import logging
import os
import pandas as pd
import time
import numpy as np
from django.conf import settings

from project_4.Cold_start_recommendation.Cold_start import (
    ColdStart,
    get_R_U_V,
    load_movie_data,
    active_learning_step
)
from .metrics_recorder import MetricsRecorder, TimingTracker

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
INITIAL_MOVIES_COUNT = 5
MAX_ACTIVE_LEARNING_ROUNDS = 5


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

    def __init__(self, session_id=None, study_mode=None):
        self.session_id = session_id or f'session_{id(self)}'
        self.study_mode = study_mode or 'unknown'

        # Initialize metrics recording
        self.metrics_recorder = MetricsRecorder(self.session_id, self.study_mode)
        self.timing_tracker = TimingTracker()

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
        self.initial_movies_count = INITIAL_MOVIES_COUNT
        self.active_learning_round = 0
        self.max_active_learning_rounds = MAX_ACTIVE_LEARNING_ROUNDS

        # Track current movie timing
        self.current_movie_start_time = None

    def start_movie_timing(self, movie_id):
        """Start timing for rating a specific movie"""
        self.current_movie_start_time = time.time()
        self.timing_tracker.start_action(f"movie_{movie_id}")

    def get_expected_rating_at_time_of_rating(self, movie_id, movie_cache):
        """
        Calculate what the predicted rating was at the time of rating submission.
        This is crucial for RMSE calculation - we need the prediction that existed
        when the user made their rating decision.
        """
        if self.user_vector is None:
            logger.info(f"No user vector available for movie {movie_id} - likely initial rating")
            return None

        try:
            # Ensure movie_id exists in the matrix
            movie_cache.R_matrix.columns = movie_cache.R_matrix.columns.astype(int)
            if movie_id not in movie_cache.R_matrix.columns:
                logger.warning(f"Movie ID {movie_id} not found in R matrix columns")
                return None

            # Get movie index and vector
            movie_index = list(movie_cache.R_matrix.columns).index(movie_id)
            movie_vector = movie_cache.V_matrix[movie_index]

            # Calculate predicted rating using current user vector
            predicted_rating = np.dot(self.user_vector, movie_vector)

            # Clip to valid rating range
            predicted_rating = float(np.clip(predicted_rating, 0.5, 5.0))

            logger.info(f"Expected rating for movie {movie_id}: {predicted_rating:.3f}")
            return predicted_rating

        except Exception as e:
            logger.error(f"Error calculating expected rating for {movie_id}: {e}")
            return None

    def store_rating(self, movie_id, rating, movie_title=None, predicted_rating=None):
        """Store a rating and record metrics"""
        # Calculate time taken if timing was started
        time_taken = None
        if self.current_movie_start_time:
            time_taken = time.time() - self.current_movie_start_time
            self.current_movie_start_time = None

        # Store in session
        rating_data = {
            'movie_id': movie_id,
            'rating': rating,
            'step': self.current_step,
            'time_taken': time_taken,
            'predicted_rating': predicted_rating
        }
        self.rated_movies.append(rating_data)
        self.total_ratings += 1
        self.current_step += 1

        # Record metrics
        phase = 'initial' if self.is_in_initial_phase() else 'active_learning'
        self.metrics_recorder.session_phase = phase
        self.metrics_recorder.record_rating(movie_id, rating, movie_title, time_taken, predicted_rating)

        logger.info(
            f"Stored rating: {movie_id} -> {rating} (expected: {predicted_rating:.3f}, time: {time_taken:.2f}s)"
            if predicted_rating and time_taken
            else f"Stored rating: {movie_id} -> {rating}")

    def get_rated_dict(self):
        """Get ratings as dictionary"""
        return {d['movie_id']: d['rating'] for d in self.rated_movies}

    def update_user_vector(self, movie_cache):
        """Update user vector with all current ratings"""
        rated_dict = self.get_rated_dict()
        cold_start = ColdStart(movie_cache.V_matrix, movie_cache.R_matrix)
        self.user_vector = cold_start.update_user_vector(rated_dict)
        logger.info(f"Updated user vector. Total ratings: {len(rated_dict)}")

    def get_predicted_rating(self, movie_id, movie_cache):
        """Calculate predicted rating for a movie given current user vector"""
        if self.user_vector is None:
            return None

        try:
            movie_cache.R_matrix.columns = movie_cache.R_matrix.columns.astype(int)
            if movie_id in movie_cache.R_matrix.columns:
                movie_index = list(movie_cache.R_matrix.columns).index(movie_id)
                movie_vector = movie_cache.V_matrix[movie_index]
                predicted_rating = np.dot(self.user_vector, movie_vector)
                return float(np.clip(predicted_rating, 0.5, 5.0))
        except Exception as e:
            logger.error(f"Error calculating predicted rating for {movie_id}: {e}")

        return None

    def finalize_session(self, movie_cache):
        """Finalize session and calculate final metrics"""
        if self.metrics_recorder and movie_cache.loaded:
            final_metrics = self.metrics_recorder.finalize_session(
                V_matrix=movie_cache.V_matrix,
                R_matrix=movie_cache.R_matrix,
                user_vector=self.user_vector
            )
            logger.info(f"Session finalized. Final metrics: {final_metrics}")
            return final_metrics
        return None

    def get_session_context(self):
        """Get current session state for responses"""
        if not self.is_initialized:
            return {}

        context = {
            'session_active': True,
            'current_step': self.current_step,
            'initial_movies_count': self.initial_movies_count,
            'total_ratings': self.total_ratings,
            'rated_count': len(self.rated_movies),
            'current_movies': self.current_movies,
            'session_phase': 'initial_rating' if self.current_step < self.initial_movies_count else 'active_learning',
            'active_learning_round': self.active_learning_round,
            'max_active_learning_rounds': self.max_active_learning_rounds
        }

        # Add current metrics
        if self.metrics_recorder:
            current_metrics = self.metrics_recorder.get_current_metrics()
            context.update(current_metrics)

        return context

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

    # Record phase change to initial
    session.metrics_recorder.set_phase('initial')

    return initial_movies


def get_next_initial_movie(session):
    """Get next movie in initial rating phase"""
    initial_movies = session.current_movies
    current_index = session.current_movie_index + 1
    session.current_movie_index = current_index

    if current_index < len(initial_movies):
        next_movie = initial_movies[current_index]
        # Start timing for this movie
        session.start_movie_timing(next_movie['movie_id'])
        return next_movie
    else:
        return None


def get_next_active_learning_movie(session, movie_cache):
    """Get next movie in active learning phase using active learning step"""
    try:
        # Use the existing active_learning_step function
        top_movie_id, top_movie_title, predicted_rating = active_learning_step(
            session.user_vector,
            movie_cache.V_matrix,
            movie_cache.R_matrix,
            session.get_rated_dict(),
            movie_cache.movieId_to_title,
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

        # Only start timing if session is not complete (for display-only movies, don't time)
        if not session.is_session_complete():
            # Start timing for this movie
            session.start_movie_timing(top_movie_id)

            logger.info(f"Active learning selected: {top_movie_title} "
                        f"(predicted: {predicted_rating:.2f})")
        else:
            logger.info(f"Final display movie: {top_movie_title} "
                        f"(predicted: {predicted_rating:.2f}) - display only")

        return movie_data

    except Exception as e:
        logger.error(f"Error in active learning step: {str(e)}")
        return None


def process_rating_submission(session, movie_cache, movie_id, rating):
    """Process rating submission and determine next action"""

    # CRITICAL: Get the expected rating BEFORE updating user vector
    # This represents what the system predicted when the user made their decision
    expected_rating = session.get_expected_rating_at_time_of_rating(movie_id, movie_cache)

    # Get movie title for metrics
    movie_title = movie_cache.movieId_to_title.get(movie_id, "Unknown")

    # Store rating with the expected value calculated at decision time
    # NOTE: store_rating() increments current_step, so do this first
    session.store_rating(movie_id, rating, movie_title, expected_rating)

    # NOW update the user vector (after we've recorded the prediction)
    session.update_user_vector(movie_cache)

    # Determine next movie and messages
    if session.is_in_initial_phase():
        # Still in initial phase
        next_movie = get_next_initial_movie(session)
        remaining = session.initial_movies_count - session.current_step
        message = f"Rating submitted! {remaining} more movies to rate."
        session_step = 'initial_rating'
    else:
        # We've just transitioned to or are continuing active learning phase
        if session.current_step == session.initial_movies_count:
            session.metrics_recorder.set_phase('active_learning')
            message = "Initial rating complete! Moving to personalized recommendations."
            session.active_learning_round = 1
        else:
            # Continuing active learning
            session.active_learning_round += 1
            round_num = session.active_learning_round
            max_rounds = session.max_active_learning_rounds
            message = f"Rating submitted! Round {round_num}/{max_rounds} of active learning."

        session_step = 'active_learning'

        next_movie = get_next_active_learning_movie(session, movie_cache)

        # Check if session should complete
        if session.is_session_complete():
            # Session is complete - get a final movie for display only
            message = "Active learning complete! Here's what we'd recommend next based on your ratings."
            session_step = 'complete'
            session.finalize_session(movie_cache)

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
