import json
import logging
import os

import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from project_4.Cold_start_recommendation.Cold_start import (
    ColdStart,
    get_R_U_V
)

DEBUG = True
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global state for active learning session
COLDSTART_STATE = {}

# Global cache for movie data
MOVIE_CACHE = {
    'movies_df': None,
    'ratings_df': None,
    'R_matrix': None,
    'V_matrix': None,  # Item features from matrix factorization
    'loaded': False
}

def index(request):
    """Main view for Cold Start Active Learning - handles both GET and POST"""

    # Handle AJAX POST requests
    if request.method == 'POST':
        return handle_ajax_request(request)

    # GET request - render initial page
    context = {
        'session_active': COLDSTART_STATE.get('is_initialized', False),
        'current_step': COLDSTART_STATE.get('current_step', 0),
        'total_ratings': COLDSTART_STATE.get('total_ratings', 0)
    }

    # Add current session context if exists
    if COLDSTART_STATE.get('is_initialized'):
        context.update(get_current_session_context())

    return render(request, 'project4_coldstart.html', context)


def handle_ajax_request(request):
    """Handle AJAX POST requests and return JSON responses"""
    try:
        data = json.loads(request.body)
        action = data.get('action')

        if DEBUG:
            logger.info(f"AJAX action: {action}")

        if action == 'initialize_session':
            return initialize_coldstart_session(data)
        elif action == 'submit_rating':
            return submit_rating(data)
        elif action == 'skip_movie':
            return skip_movie(data)
        elif action == 'reset_session':
            return reset_session()
        elif action == 'get_rating_explanation':
            return get_rating_explanation(data)
        else:
            return JsonResponse({'error': f'Unknown action: {action}'}, status=400)

    except Exception as e:
        logger.error(f"Error handling AJAX request: {str(e)}")
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


def initialize_coldstart_session(data, n=8):
    """Initialize cold start session with initial movie selection"""
    global COLDSTART_STATE

    try:
        # Ensure movie data is loaded
        ensure_movie_data_loaded()
        df_movies = MOVIE_CACHE['movies_df']
        movieId_to_title = dict(zip(df_movies['movieId'], df_movies['title']))


        # Getting the n initial movies to rate
        raw_movies = get_initial_movie_selection(n)

        # Re-formatting them
        initial_movies = [
            {
                'movie_id': m['movieId'],
                'title': m['title'],
                'genres': m['genres'],
            }
            for m in raw_movies
        ]


        # Initialize user state
        COLDSTART_STATE = {
            'is_initialized': True,
            'current_step': 0,
            'total_ratings': 0,
            'user_vector': None,  # Will be initialized after first ratings
            'current_movies': initial_movies,
            'current_movie_index': 0,
            'rated_movies': [],
            'skipped_movies': [],
            'session_id': data.get('session_id', 'default_session')
        }

        response_data = {
            'message': 'Cold start session initialized successfully',
            'success': True,
            'movies': initial_movies,
            'session_step': 'initial_rating'
        }
        response_data.update(get_current_session_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error initializing cold start session: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def submit_rating(data, n=8):
    """Process user rating and update user vector"""
    global COLDSTART_STATE

    try:
        if not COLDSTART_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Session not initialized'}, status=400)

        movie_id = data.get('movie_id')
        rating = float(data.get('rating'))

        if not movie_id or rating is None:
            return JsonResponse({'error': 'Movie ID and rating required'}, status=400)

        # Store the rating
        rating_data = {
            'movie_id': movie_id,
            'rating': rating,
            'step': COLDSTART_STATE['current_step']
        }
        COLDSTART_STATE['rated_movies'].append(rating_data)
        COLDSTART_STATE['total_ratings'] += 1

        # Dictionary of all rated movies so far
        rated_dict = {d['movie_id']: d['rating'] for d in COLDSTART_STATE['rated_movies']}

        # Cold start active learning function
        cold_start = ColdStart(MOVIE_CACHE['V_matrix'], MOVIE_CACHE['R_matrix'])
        updated_user_vector = cold_start.update_user_vector(rated_dict)
        COLDSTART_STATE['user_vector'] = updated_user_vector

        # Increment step
        COLDSTART_STATE['current_step'] += 1

        # Initialize current_movie_index if it doesn't exist
        if 'current_movie_index' not in COLDSTART_STATE:
            COLDSTART_STATE['current_movie_index'] = 0

        COLDSTART_STATE['current_movie_index'] += 1

        # Check if we're in initial rating phase (first n movies)
        if COLDSTART_STATE['current_step'] < n:
            session_step = 'initial_rating'
            remaining = n - COLDSTART_STATE['current_step']
            message = f"Rating submitted! {remaining} more movies to rate."

            # Get next movie from initial movies list
            initial_movies = COLDSTART_STATE.get('current_movies', [])
            current_index = COLDSTART_STATE.get('current_movie_index', 0)

            if current_index < len(initial_movies):
                next_movie = initial_movies[current_index]
            else:
                # Fallback: get recommendation if we run out of initial movies
                next_movie = get_next_movie_recommendation(
                    COLDSTART_STATE['user_vector'],
                    COLDSTART_STATE['rated_movies'],
                    COLDSTART_STATE['skipped_movies'])

        else:
            # Move to active learning phase
            session_step = 'active_learning'
            if COLDSTART_STATE['current_step'] == n:
                message = f"Initial rating complete! Moving to personalized recommendations."
            else:
                message = "Rating submitted! Getting your next recommendation..."

            # Get recommendation using algorithm
            next_movie = get_next_movie_recommendation(
                COLDSTART_STATE['user_vector'],
                COLDSTART_STATE['rated_movies'],
                COLDSTART_STATE['skipped_movies'])

        response_data = {
            'message': message,
            'success': True,
            'session_step': session_step,
            'next_movie': next_movie
        }
        response_data.update(get_current_session_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error submitting rating: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def skip_movie(data):
    """Skip current movie without affecting user vector"""
    global COLDSTART_STATE

    try:
        if not COLDSTART_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Session not initialized'}, status=400)

        movie_id = data.get('movie_id')
        if not movie_id:
            return JsonResponse({'error': 'Movie ID required'}, status=400)

        # Store the skip
        skip_data = {
            'movie_id': movie_id,
            'step': COLDSTART_STATE['current_step']
        }
        COLDSTART_STATE['skipped_movies'].append(skip_data)

        next_movie = get_replacement_movie(
            COLDSTART_STATE['user_vector'],
            COLDSTART_STATE['rated_movies'],
            COLDSTART_STATE['skipped_movies']
        )

        if next_movie:
            COLDSTART_STATE['current_movies'] = [next_movie]
            message = "Movie skipped. Here's another recommendation for you."
        else:
            message = "Movie skipped. No more recommendations available."
            COLDSTART_STATE['current_movies'] = []

        response_data = {
            'message': message,
            'success': True,
            'next_movie': next_movie
        }
        response_data.update(get_current_session_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error skipping movie: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def get_rating_explanation(data):
    """Get explanation of how different ratings would affect future recommendations"""
    global COLDSTART_STATE

    try:
        if not COLDSTART_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Session not initialized'}, status=400)

        movie_id = data.get('movie_id')
        if not movie_id:
            return JsonResponse({'error': 'Movie ID required'}, status=400)

        # Get current user ratings
        current_ratings = {d['movie_id']: d['rating'] for d in COLDSTART_STATE['rated_movies']}

        # Generate explanations for each possible rating (1-5)
        explanations = []

        for hypothetical_rating in range(1, 6):
            # TODO: Implement the actual explanation logic
            # This should analyze how the rating would affect:
            # 1. User's latent feature vector
            # 2. Future movie recommendations
            # 3. Which features/preferences would be emphasized

            explanation = generate_rating_impact_explanation(
                current_ratings,
                movie_id,
                hypothetical_rating,
                MOVIE_CACHE['V_matrix'],
                MOVIE_CACHE['R_matrix']
            )

            explanations.append({
                'rating': hypothetical_rating,
                'explanation': explanation
            })

        response_data = {
            'success': True,
            'movie_id': movie_id,
            'explanations': explanations
        }

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error getting rating explanation: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def generate_rating_impact_explanation(current_ratings, movie_id, rating, V_matrix, R_matrix):
    """
    Generate explanation of how a specific rating would impact the user's profile and future recommendations

    Args:
        current_ratings: Dictionary of current user ratings {movie_id: rating}
        movie_id: ID of the movie being rated
        rating: The hypothetical rating (1-5)
        V_matrix: Movie feature matrix from matrix factorization
        R_matrix: Original rating matrix

    Returns:
        Dictionary containing explanation details
    """
    # TODO: Implement the actual explanation generation logic
    # This is a template function that should be implemented with the actual algorithm

    # Placeholder explanation structure
    explanation = {
        'predicted_next_movie': {
            'title': 'Example Movie Title',
            'confidence': 0.85,
            'reason': 'Based on your updated preferences'
        },
        'feature_changes': [
            {
                'feature_name': 'Action Adventure',
                'change': 'increased',
                'magnitude': 0.3,
                'description': 'Your preference for action movies would increase'
            },
            {
                'feature_name': 'Romantic Comedy',
                'change': 'decreased',
                'magnitude': 0.1,
                'description': 'Your preference for romantic comedies would slightly decrease'
            }
        ],
        'similarity_to_users': {
            'most_similar_user_type': 'Action movie enthusiasts',
            'similarity_score': 0.72
        },
        'recommendation_confidence': 0.78
    }

    return explanation


def reset_session():
    """Reset the cold start session"""
    global COLDSTART_STATE

    try:
        COLDSTART_STATE = {}
        return JsonResponse({
            'message': 'Cold start session reset successfully',
            'success': True
        })
    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def get_current_session_context():
    """Get current session state for JSON response"""
    if not COLDSTART_STATE.get('is_initialized'):
        return {}

    return {
        'session_active': True,
        'current_step': COLDSTART_STATE.get('current_step', 0),
        'total_ratings': COLDSTART_STATE.get('total_ratings', 0),
        'rated_count': len(COLDSTART_STATE.get('rated_movies', [])),
        'skipped_count': len(COLDSTART_STATE.get('skipped_movies', [])),
        'current_movies': COLDSTART_STATE.get('current_movies', []),
        'session_phase': 'initial_rating' if COLDSTART_STATE.get('current_step', 0) < 10 else 'active_learning'
    }


def ensure_movie_data_loaded():
    """Load movie dataset and matrix factorization results"""
    global MOVIE_CACHE

    if MOVIE_CACHE['loaded']:
        return

    try:
        # Load movie and rating data
        movies_path = os.path.join(settings.BASE_DIR, 'data', 'ml_latest_small', 'movies.csv')
        ratings_path = os.path.join(settings.BASE_DIR, 'data', 'ml_latest_small', 'ratings.csv')

        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        model, R_matrix, U_matrix, V_matrix = get_R_U_V()

        MOVIE_CACHE.update({
            'movies_df': movies_df,
            'ratings_df': ratings_df,
            'R_matrix': R_matrix,
            'V_matrix': V_matrix,
            'loaded': True,
            'selected_movies': None
        })

        if DEBUG:
            logger.info("Movie data loaded successfully")

    except Exception as e:
        logger.error(f"Error loading movie data: {str(e)}")
        raise


def get_initial_movie_selection(n=5):
    if MOVIE_CACHE.get('selected_movies') is None:
        # Import here to avoid circular imports
        from project_4.Cold_start_recommendation.Clustering import run_true_hybrid_cold_start

        # Use already loaded data from cache
        df_movies = MOVIE_CACHE['movies_df']
        df_ratings = MOVIE_CACHE['ratings_df']
        R_matrix = MOVIE_CACHE['R_matrix']
        V_matrix = MOVIE_CACHE['V_matrix']

        # Create a mock model object with V matrix
        class MockModel:
            def __init__(self, V):
                self.V = V

        mock_model = MockModel(V_matrix)

        # Get selected movies using already loaded data
        selected_movies, _, _, _ = run_true_hybrid_cold_start(
            df_movies=df_movies,
            df_ratings=df_ratings,
            mat_fac_model=mock_model,
            R_matrix=R_matrix,
            n_clusters=10,
            genre_weight=0.7,
            latent_weight=0.3,
            top_k_candidates=3
        )

        MOVIE_CACHE['selected_movies'] = selected_movies

    import random
    selected_movies = MOVIE_CACHE['selected_movies']
    # Use a fixed seed or cache the sampled movies
    if 'sampled_movies' not in MOVIE_CACHE:
        MOVIE_CACHE['sampled_movies'] = random.sample(selected_movies, min(n, len(selected_movies)))
    return MOVIE_CACHE['sampled_movies']


def update_user_vector(rated_dict):
    cold_start = ColdStart(MOVIE_CACHE['V_matrix'],MOVIE_CACHE['R_matrix'])
    updated_vector = cold_start.update_user_vector(rated_dict)
    return updated_vector

def get_next_movie_recommendation(user_vector, rated_movies, skipped_movies):
    rated_ids = {m['movie_id'] for m in rated_movies}
    skipped_ids = {m['movie_id'] for m in skipped_movies}
    excluded_ids = rated_ids.union(skipped_ids)


    R = MOVIE_CACHE['R_matrix']
    V = MOVIE_CACHE['V_matrix']
    movie_ids_list = list(R.columns)

    candidate_ids = [i for i in R.columns if i not in excluded_ids]
    if not candidate_ids:
        return None

    candidate_indices = [movie_ids_list.index(i) for i in candidate_ids]
    V_candidates = V[candidate_indices,:]
    predicted_ratings = V_candidates @ user_vector

    top_idx = np.argmax(predicted_ratings)
    best_movie_id = candidate_ids[top_idx]
    predicted_score = predicted_ratings[top_idx]

    movie_info = MOVIE_CACHE['movies_df'][MOVIE_CACHE['movies_df']['movieId'] == best_movie_id].iloc[0]

    next_movie = {
        'movie_id': int(best_movie_id),
        'title': movie_info['title'],
        'genres': movie_info['genres'],
        'predicted_rating': float(predicted_score)
    }

    return next_movie

def get_replacement_movie(user_vector, rated_movies, skipped_movies):
    """
    TODO: Get replacement movie when user skips
    This should find an alternative movie that doesn't affect the model
    or user vector, excluding already rated/skipped movies
    """
    # Placeholder implementation
    if DEBUG:
        logger.info("Getting replacement movie after skip")

    # TODO: Implement replacement logic
    # Should exclude movies in rated_movies and skipped_movies
    replacement_movie = {
        'movie_id': 1000,
        'title': 'Alternative Movie',
        'genres': 'Comedy|Family',
        'year': 2019,
        'poster_url': None
    }

    return replacement_movie