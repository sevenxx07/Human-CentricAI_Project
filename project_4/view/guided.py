import json
import logging
import os
import random

import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from project_4.Cold_start_recommendation.Cold_start import (
    ColdStart,
    get_R_U_V,
    load_movie_data, active_learning_step
)
from project_4.Cold_start_recommendation.feature_interpretations import (
    feature_dict,
    feature_characteristics
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global state management
COLDSTART_STATE = {}
MOVIE_CACHE = {
    'movies_df': None,
    'ratings_df': None,
    'R_matrix': None,
    'V_matrix': None,
    'movieId_to_title': None,
    'loaded': False,
    'selected_movies': None,
    'sampled_movies': None
}

# Configuration constants
INITIAL_MOVIES_COUNT = 5
MAX_ACTIVE_LEARNING_ROUNDS = 20


# ==================== MAIN VIEW HANDLERS ====================

def index(request):
    """Main view for Cold Start Active Learning"""
    if request.method == 'POST':
        return handle_ajax_request(request)

    context = {
        'session_active': COLDSTART_STATE.get('is_initialized', False),
        'current_step': COLDSTART_STATE.get('current_step', 0),
        'total_ratings': COLDSTART_STATE.get('total_ratings', 0)
    }

    if COLDSTART_STATE.get('is_initialized'):
        context.update(get_current_session_context())

    return render(request, 'project4_guided.html', context)


def handle_ajax_request(request):
    """Route AJAX requests to appropriate handlers"""
    try:
        data = json.loads(request.body)
        action = data.get('action')
        logger.info(f"AJAX action: {action}")

        handlers = {
            'initialize_session': initialize_coldstart_session,
            'submit_rating': submit_rating,
            'skip_movie': skip_movie,
            'reset_session': reset_session,
            'get_rating_explanation': get_rating_explanation
        }

        if action in handlers:
            return handlers[action](data)
        else:
            return JsonResponse({'error': f'Unknown action: {action}'}, status=400)

    except Exception as e:
        logger.error(f"Error handling AJAX request: {str(e)}")
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


# ==================== SESSION MANAGEMENT ====================

def initialize_coldstart_session(data):
    """Initialize cold start session"""
    global COLDSTART_STATE

    try:
        ensure_movie_data_loaded()
        raw_movies = get_initial_movie_selection(INITIAL_MOVIES_COUNT)

        initial_movies = [
            {
                'movie_id': m['movieId'],
                'title': m['title'],
                'genres': m['genres'],
            }
            for m in raw_movies
        ]

        COLDSTART_STATE = {
            'is_initialized': True,
            'current_step': 0,
            'total_ratings': 0,
            'user_vector': None,
            'current_movies': initial_movies,
            'current_movie_index': 0,
            'rated_movies': [],
            'skipped_movies': [],
            'session_id': data.get('session_id', 'default_session'),
            'initial_movies_count': INITIAL_MOVIES_COUNT,
            'active_learning_round': 0,
            'max_active_learning_rounds': MAX_ACTIVE_LEARNING_ROUNDS
        }

        first_movie = initial_movies[0] if initial_movies else None

        response_data = {
            'message': 'Cold start session initialized successfully',
            'success': True,
            'movies': initial_movies,
            'session_step': 'initial_rating',
            'next_movie': first_movie
        }
        response_data.update(get_current_session_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error initializing cold start session: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def reset_session(data=None):
    """Reset the cold start session"""
    global COLDSTART_STATE

    try:
        COLDSTART_STATE = {}
        # Clear cached selections but keep expensive data loaded
        MOVIE_CACHE['selected_movies'] = None
        MOVIE_CACHE['sampled_movies'] = None

        return JsonResponse({
            'message': 'Cold start session reset successfully',
            'success': True
        })
    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def get_current_session_context():
    """Get current session state for responses"""
    if not COLDSTART_STATE.get('is_initialized'):
        return {}

    initial_count = COLDSTART_STATE.get('initial_movies_count')
    current_step = COLDSTART_STATE.get('current_step', 0)
    active_round = COLDSTART_STATE.get('active_learning_round', 0)

    return {
        'session_active': True,
        'current_step': current_step,
        'initial_movies_count': initial_count,
        'total_ratings': COLDSTART_STATE.get('total_ratings', 0),
        'rated_count': len(COLDSTART_STATE.get('rated_movies', [])),
        'skipped_count': len(COLDSTART_STATE.get('skipped_movies', [])),
        'current_movies': COLDSTART_STATE.get('current_movies', []),
        'session_phase': 'initial_rating' if current_step < initial_count else 'active_learning',
        'active_learning_round': active_round,
        'max_active_learning_rounds': COLDSTART_STATE.get('max_active_learning_rounds', MAX_ACTIVE_LEARNING_ROUNDS)
    }


# ==================== RATING HANDLERS ====================

def submit_rating(data):
    """Process user rating and update user vector"""
    global COLDSTART_STATE

    try:
        if not COLDSTART_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Session not initialized'}, status=400)

        movie_id = data.get('movie_id')
        rating = float(data.get('rating'))

        if not movie_id or rating is None:
            return JsonResponse({'error': 'Movie ID and rating required'}, status=400)

        # Store rating and update user vector
        store_rating(movie_id, rating)
        update_user_vector()

        # Determine next action
        next_movie, message, session_step = get_next_action()

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

        # Store skip
        skip_data = {
            'movie_id': movie_id,
            'step': COLDSTART_STATE['current_step']
        }
        COLDSTART_STATE['skipped_movies'].append(skip_data)

        # Get next movie
        next_movie, message, session_step = get_next_movie_after_skip()

        response_data = {
            'message': message,
            'success': True,
            'next_movie': next_movie,
            'session_step': session_step
        }
        response_data.update(get_current_session_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error skipping movie: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def store_rating(movie_id, rating):
    """Store a rating and increment counters"""
    rating_data = {
        'movie_id': movie_id,
        'rating': rating,
        'step': COLDSTART_STATE['current_step']
    }
    COLDSTART_STATE['rated_movies'].append(rating_data)
    COLDSTART_STATE['total_ratings'] += 1
    COLDSTART_STATE['current_step'] += 1

    logger.info(f"Stored rating: {movie_id} -> {rating}")


def update_user_vector():
    """Update user vector with all current ratings"""
    rated_dict = {d['movie_id']: d['rating'] for d in COLDSTART_STATE['rated_movies']}
    cold_start = ColdStart(MOVIE_CACHE['V_matrix'], MOVIE_CACHE['R_matrix'])
    updated_user_vector = cold_start.update_user_vector(rated_dict)
    COLDSTART_STATE['user_vector'] = updated_user_vector

    logger.info(f"Updated user vector. Total ratings: {len(rated_dict)}")


# ==================== MOVIE SELECTION LOGIC ====================

def get_next_action():
    """Determine next movie and phase after rating submission"""
    initial_count = COLDSTART_STATE.get('initial_movies_count')
    current_step = COLDSTART_STATE.get('current_step')

    if current_step < initial_count:
        return get_next_initial_movie()
    else:
        return get_next_active_learning_movie()


def get_next_initial_movie():
    """Get next movie in initial rating phase"""
    initial_count = COLDSTART_STATE.get('initial_movies_count')
    current_step = COLDSTART_STATE.get('current_step')
    remaining = initial_count - current_step

    message = f"Rating submitted! {remaining} more movies to rate."
    session_step = 'initial_rating'

    # Get next movie from initial list
    initial_movies = COLDSTART_STATE.get('current_movies', [])
    current_index = COLDSTART_STATE.get('current_movie_index', 0) + 1
    COLDSTART_STATE['current_movie_index'] = current_index

    if current_index < len(initial_movies):
        next_movie = initial_movies[current_index]
    else:
        next_movie = None

    return next_movie, message, session_step


def get_next_active_learning_movie():
    """Get next movie in active learning phase"""
    initial_count = COLDSTART_STATE.get('initial_movies_count')
    current_step = COLDSTART_STATE.get('current_step')

    if current_step == initial_count:
        message = "Initial rating complete! Moving to personalized recommendations."
        COLDSTART_STATE['active_learning_round'] = 1
    else:
        COLDSTART_STATE['active_learning_round'] += 1
        round_num = COLDSTART_STATE['active_learning_round']
        max_rounds = COLDSTART_STATE.get('max_active_learning_rounds')
        message = f"Rating submitted! Round {round_num}/{max_rounds} of active learning."

    session_step = 'active_learning'

    # Check if session should continue
    if COLDSTART_STATE['active_learning_round'] <= COLDSTART_STATE.get('max_active_learning_rounds'):
        next_movie = run_active_learning_step(
            COLDSTART_STATE['user_vector'],
            {d['movie_id']: d['rating'] for d in COLDSTART_STATE['rated_movies']},
            COLDSTART_STATE['skipped_movies']
        )
    else:
        message = "Active learning session complete! Thank you for your participation."
        next_movie = None

    return next_movie, message, session_step


def get_next_movie_after_skip():
    """Handle movie selection after skip"""
    initial_count = COLDSTART_STATE.get('initial_movies_count')
    current_step = COLDSTART_STATE.get('current_step', 0)

    if current_step < initial_count:
        return handle_skip_in_initial_phase()
    else:
        return active_learning_step(
            COLDSTART_STATE['user_vector'],
            {d['movie_id']: d['rating'] for d in COLDSTART_STATE['rated_movies']},
            COLDSTART_STATE['skipped_movies']
        )


def handle_skip_in_initial_phase():
    """Handle skip during initial rating phase"""
    message = "Movie skipped. Continue with initial rating."
    session_phase = 'initial_rating'

    initial_movies = COLDSTART_STATE.get('current_movies', [])
    current_index = COLDSTART_STATE.get('current_movie_index', 0) + 1
    COLDSTART_STATE['current_movie_index'] = current_index

    #TODO something from the same cluster
    if current_index < len(initial_movies):
        next_movie = initial_movies[current_index]
    else:
        next_movie = None
        message = "Initial movies completed."

    return next_movie, message, session_phase


def run_active_learning_step(user_vector, user_ratings_dict, skipped_movies):
    """Select the most informative movie using active learning with existing clustering logic"""
    try:
        from project_4.Cold_start_recommendation.Cold_start import active_learning_step

        # Get raw data from the existing active_learning_step function
        top_movie_id, top_movie_title, predicted_rating = active_learning_step(
            user_vector,
            MOVIE_CACHE['V_matrix'],
            MOVIE_CACHE['R_matrix'],
            user_ratings_dict,
            MOVIE_CACHE['movieId_to_title'],
            skipped_movies
        )

        # Get additional movie info (genres) from movies dataframe
        movie_info = MOVIE_CACHE['movies_df'][
            MOVIE_CACHE['movies_df']['movieId'] == top_movie_id
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


# ==================== EXPLANATION SYSTEM ====================

def get_rating_explanation(data):
    """Generate explanations for different rating impacts using Cold_start functions"""
    try:
        if not COLDSTART_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Session not initialized'}, status=400)

        movie_id = int(data.get('movie_id'))
        if not movie_id:
            return JsonResponse({'error': 'Movie ID required'}, status=400)

        current_ratings = {d['movie_id']: d['rating'] for d in COLDSTART_STATE.get('rated_movies', [])}

        # Use the existing explain_impact function from Cold_start.py
        from project_4.Cold_start_recommendation.Cold_start import explain_impact

        raw_explanations = explain_impact(
            current_ratings,
            movie_id,
            MOVIE_CACHE['V_matrix'],
            MOVIE_CACHE['R_matrix'],
            MOVIE_CACHE['movieId_to_title']
        )

        # Transform raw data to what GUI expects
        formatted_explanations = []
        for rating, next_movie_id, next_movie_title, next_predicted_rating, feature_changes in raw_explanations:

            # Transform feature changes
            transformed_features = []
            for feature_title, direction, feature_info, user_val, movie_val, match in feature_changes:
                transformed_features.append({
                    'feature_name': feature_title,
                    'change': 'increase' if direction == 'increased' else 'decrease',
                    'description': f"{direction.capitalize()} your preference for {feature_info.lower()}"
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

        return JsonResponse({
            'success': True,
            'movie_id': movie_id,
            'explanations': formatted_explanations
        })

    except Exception as e:
        logger.error(f"Error getting rating explanation: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


# ==================== DATA MANAGEMENT ====================

def ensure_movie_data_loaded():
    """Load movie dataset and matrix factorization results"""
    global MOVIE_CACHE

    if MOVIE_CACHE['loaded']:
        return

    try:
        movies_path = os.path.join(settings.BASE_DIR, 'data', 'ml_latest_small', 'movies.csv')
        ratings_path = os.path.join(settings.BASE_DIR, 'data', 'ml_latest_small', 'ratings.csv')

        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        model, R_matrix, U_matrix, V_matrix = get_R_U_V()
        movieId_to_title = load_movie_data()

        MOVIE_CACHE.update({
            'movies_df': movies_df,
            'ratings_df': ratings_df,
            'R_matrix': R_matrix,
            'V_matrix': V_matrix,
            'movieId_to_title': movieId_to_title,
            'loaded': True
        })

        logger.info("Movie data loaded successfully")

    except Exception as e:
        logger.error(f"Error loading movie data: {str(e)}")
        raise


def get_initial_movie_selection(n):
    """Get initial movies using clustering"""
    if MOVIE_CACHE.get('selected_movies') is None:
        from project_4.Cold_start_recommendation.Clustering import run_true_hybrid_cold_start

        class MockModel:
            def __init__(self, V):
                self.V = V

        mock_model = MockModel(MOVIE_CACHE['V_matrix'])

        selected_movies, _, _, _ = run_true_hybrid_cold_start(
            df_movies=MOVIE_CACHE['movies_df'],
            df_ratings=MOVIE_CACHE['ratings_df'],
            mat_fac_model=mock_model,
            R_matrix=MOVIE_CACHE['R_matrix'],
            n_clusters=n
        )

        MOVIE_CACHE['selected_movies'] = selected_movies

    return MOVIE_CACHE['selected_movies']
