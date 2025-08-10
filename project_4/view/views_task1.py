import json
import logging
import os

from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings

import pandas as pd 
import numpy as np


from project_4.Cold_start_recommendation.Cold_start import (
    ColdStart, 
    get_initial_movies,
    get_initial_ratings, 
    active_learning_loop, 
    get_R_U_V,
    get_selected_cold_start_movies, 
    feature_characteristics,
    feature_dict,
    run_cold_start_demo
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
        raw_movies = get_initial_movies(5)
        
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

        # Check if we're in initial rating phase (first n movies)
        if COLDSTART_STATE['current_step'] < n:
            COLDSTART_STATE['current_step'] += 1

            # If we've rated all initial movies, move to active learning phase
            if COLDSTART_STATE['current_step'] >= n:
                session_step = 'active_learning'
                message = f"Initial rating complete! Moving to personalized recommendations."
            else:
                session_step = 'initial_rating'
                remaining = 10 - COLDSTART_STATE['current_step']
                message = f"Rating submitted! {remaining} more movies to rate."
        else:
            # In active learning phase
            session_step = 'active_learning'
            message = "Rating submitted! Getting your next recommendation..."

        # Get next movie recommendation
        next_movie = get_next_movie_recommendation(
            COLDSTART_STATE['user_vector'],
            COLDSTART_STATE['rated_movies'],
            COLDSTART_STATE['skipped_movies'])
        
        COLDSTART_STATE['current_movies'] = [next_movie] if next_movie else []

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
            'loaded': True
        })

        if DEBUG:
            logger.info("Movie data loaded successfully")

    except Exception as e:
        logger.error(f"Error loading movie data: {str(e)}")
        raise


def get_initial_movie_selection(n=5):
    return get_initial_movies(n)
    
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