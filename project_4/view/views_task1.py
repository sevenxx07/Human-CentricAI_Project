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
    get_R_U_V,
    load_movie_data
)
from project_4.Cold_start_recommendation.feature_interpretations import (
    feature_dict,
    feature_characteristics
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
    'V_matrix': None,
    'movieId_to_title': None,
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
            # return get_rating_explanation(data)
            pass
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

        # Check if we're in initial rating phase (first n movies)
        if COLDSTART_STATE['current_step'] < n:
            session_step = 'initial_rating'
            remaining = n - COLDSTART_STATE['current_step']
            message = f"Rating submitted! {remaining} more movies to rate."

            # Get next movie from initial movies list
            initial_movies = COLDSTART_STATE.get('current_movies', [])
            current_index = COLDSTART_STATE.get('current_movie_index', 0) + 1
            COLDSTART_STATE['current_movie_index'] = current_index

            if current_index < len(initial_movies):
                next_movie = initial_movies[current_index]
            else:
                # Fallback if we run out of initial movies
                next_movie = get_next_movie_recommendation(
                    COLDSTART_STATE['user_vector'],
                    COLDSTART_STATE['rated_movies'],
                    COLDSTART_STATE['skipped_movies'])
                if next_movie:
                    message = "Moving to personalized recommendations!"
                    session_step = 'active_learning'

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
    Uses the actual feature interpretation system from your code
    """
    try:
        # Simulate the rating
        simulated_ratings = current_ratings.copy()
        simulated_ratings[movie_id] = rating

        # Get current user vector if we have ratings
        if current_ratings:
            cold_start = ColdStart(V_matrix, R_matrix)
            current_user_vector = cold_start.update_user_vector(current_ratings)
        else:
            current_user_vector = np.zeros(V_matrix.shape[1])

        # Get updated user vector with the new rating
        cold_start = ColdStart(V_matrix, R_matrix)
        updated_user_vector = cold_start.update_user_vector(simulated_ratings)

        # Calculate feature changes
        feature_deltas = updated_user_vector - current_user_vector

        # Get predicted ratings for all movies
        predicted_ratings = V_matrix @ updated_user_vector
        predicted_ratings = np.clip(predicted_ratings, 0, 5)

        # Find best unrated movie
        rated_ids = set(simulated_ratings.keys())
        movie_ids_list = list(R_matrix.columns)
        unrated_indices = [idx for idx, movie_id in enumerate(movie_ids_list) if movie_id not in rated_ids]

        if unrated_indices:
            # Get the top recommended movie
            top_index = max(unrated_indices, key=lambda i: predicted_ratings[i])
            top_movie_id = movie_ids_list[top_index]
            top_movie_title = MOVIE_CACHE['movieId_to_title'].get(top_movie_id, "Unknown Movie")
            top_movie_vector = V_matrix[top_index]
            predicted_score = predicted_ratings[top_index]
        else:
            top_movie_title = "No recommendations available"
            predicted_score = 0.0
            top_movie_vector = np.zeros(V_matrix.shape[1])

        # Analyze top feature changes
        top_feature_indices = np.argsort(np.abs(feature_deltas))[::-1][:3]
        feature_changes = []

        for idx in top_feature_indices:
            if abs(feature_deltas[idx]) < 0.01:  # Skip very small changes
                continue

            direction = "increased" if feature_deltas[idx] > 0 else "decreased"
            feature_key = f'Feature_{idx + 1}'
            feature_name = feature_dict.get(feature_key, f"Feature {idx + 1}")
            feature_description = feature_characteristics.get(feature_key, "No description available")

            # Check if this change aligns with the recommended movie
            movie_feature_value = top_movie_vector[idx]
            alignment = "strongly aligns" if np.sign(feature_deltas[idx]) == np.sign(movie_feature_value) and abs(
                movie_feature_value) > 0.1 else "differs"

            change_description = f"Your preference for '{feature_description.lower()}' would be {direction}, which {alignment} with the recommended movie"

            feature_changes.append({
                'feature_name': feature_name,
                'change': direction,
                'magnitude': abs(feature_deltas[idx]),
                'description': change_description
            })

        # Calculate confidence based on prediction strength
        confidence = min(0.95, max(0.5, (predicted_score / 5.0) * 0.8 + 0.2))

        # Determine user similarity type based on strongest features
        strongest_features = np.argsort(np.abs(updated_user_vector))[::-1][:2]
        user_types = []
        for feat_idx in strongest_features:
            feature_key = f'Feature_{feat_idx + 1}'
            if feature_key in feature_dict:
                user_types.append(feature_dict[feature_key])

        user_type = " & ".join(user_types[:2]) if user_types else "General movie enthusiasts"
        similarity_score = confidence * 0.9  # Approximate similarity

        explanation = {
            'predicted_next_movie': {
                'title': top_movie_title,
                'confidence': confidence,
                'reason': f'Based on your updated preferences for {feature_changes[0]["feature_name"] if feature_changes else "various features"}'
            },
            'feature_changes': feature_changes,
            'similarity_to_users': {
                'most_similar_user_type': user_type,
                'similarity_score': similarity_score
            },
            'recommendation_confidence': confidence
        }

        return explanation

    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        # Return fallback explanation
        return {
            'predicted_next_movie': {
                'title': 'Unable to generate prediction',
                'confidence': 0.5,
                'reason': 'Calculation error occurred'
            },
            'feature_changes': [
                {
                    'feature_name': 'Unknown',
                    'change': 'unknown',
                    'magnitude': 0.0,
                    'description': 'Unable to calculate feature changes'
                }
            ],
            'similarity_to_users': {
                'most_similar_user_type': 'General users',
                'similarity_score': 0.5
            },
            'recommendation_confidence': 0.5
        }


def reset_session():
    """Reset the cold start session"""
    global COLDSTART_STATE
    global MOVIE_CACHE

    try:
        COLDSTART_STATE = {}
        MOVIE_CACHE = {
            'movies_df': None,
            'ratings_df': None,
            'R_matrix': None,
            'V_matrix': None,
            'movieId_to_title': None,
            'loaded': False
        }

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

        # Load movie title mapping
        movieId_to_title = load_movie_data()

        MOVIE_CACHE.update({
            'movies_df': movies_df,
            'ratings_df': ratings_df,
            'R_matrix': R_matrix,
            'V_matrix': V_matrix,
            'movieId_to_title': movieId_to_title,
            'loaded': True,
            'selected_movies': None
        })

        if DEBUG:
            logger.info("Movie data loaded successfully")

    except Exception as e:
        logger.error(f"Error loading movie data: {str(e)}")
        raise


def get_initial_movie_selection(n=8):
    if MOVIE_CACHE.get('selected_movies') is None:
        # Import here to avoid circular imports
        from project_4.Cold_start_recommendation.Clustering import run_true_hybrid_cold_start

        # Use already loaded data from cache
        df_movies = MOVIE_CACHE['movies_df']
        df_ratings = MOVIE_CACHE['ratings_df']
        R_matrix = MOVIE_CACHE['R_matrix']
        V_matrix = MOVIE_CACHE['V_matrix']

        # Mock model object with V matrix
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
            n_clusters=n
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
    Get replacement movie when user skips
    This should find an alternative movie that doesn't affect the model
    or user vector, excluding already rated/skipped movies
    """
    # For now, use the same logic as get_next_movie_recommendation
    # but we could implement different logic for replacements
    return get_next_movie_recommendation(user_vector, rated_movies, skipped_movies)