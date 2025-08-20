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
    if request.method == 'POST':
        return handle_ajax_request(request)

    # GET request - render initial page
    context = {
        'session_active': COLDSTART_STATE.get('is_initialized', False),
        'current_step': COLDSTART_STATE.get('current_step', 0),
        'total_ratings': COLDSTART_STATE.get('total_ratings', 0)
    }

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
            initial_count = data.get('initial_movies_count', 8)
            return initialize_coldstart_session(data, initial_count)
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


def initialize_coldstart_session(data, initial_movies_count):
    """Initialize cold start session with initial movie selection"""
    global COLDSTART_STATE

    try:
        ensure_movie_data_loaded()
        raw_movies = get_initial_movie_selection(initial_movies_count)

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
            'initial_movies_count': initial_movies_count
        }

        # Get the first movie to show
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

        # Store the rating
        rating_data = {
            'movie_id': movie_id,
            'rating': rating,
            'step': COLDSTART_STATE['current_step']
        }
        COLDSTART_STATE['rated_movies'].append(rating_data)
        COLDSTART_STATE['total_ratings'] += 1

        # Update user vector
        rated_dict = {d['movie_id']: d['rating'] for d in COLDSTART_STATE['rated_movies']}
        cold_start = ColdStart(MOVIE_CACHE['V_matrix'], MOVIE_CACHE['R_matrix'])
        updated_user_vector = cold_start.update_user_vector(rated_dict)
        COLDSTART_STATE['user_vector'] = updated_user_vector

        # Increment step
        COLDSTART_STATE['current_step'] += 1

        # Determine next action based on phase
        initial_count = COLDSTART_STATE.get('initial_movies_count', 8)
        next_movie = None

        if COLDSTART_STATE['current_step'] < initial_count:
            # Still in initial rating phase
            session_step = 'initial_rating'
            remaining = initial_count - COLDSTART_STATE['current_step']
            message = f"Rating submitted! {remaining} more movies to rate."

            # Get next movie from initial movies list
            initial_movies = COLDSTART_STATE.get('current_movies', [])
            current_index = COLDSTART_STATE.get('current_movie_index', 0) + 1
            COLDSTART_STATE['current_movie_index'] = current_index

            if current_index < len(initial_movies):
                next_movie = initial_movies[current_index]
        else:
            # Move to active learning phase
            session_step = 'active_learning'
            if COLDSTART_STATE['current_step'] == initial_count:
                message = "Initial rating complete! Moving to personalized recommendations."
            else:
                message = "Rating submitted! Getting your next recommendation..."

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

        # Determine current phase
        initial_count = COLDSTART_STATE.get('initial_movies_count', 8)
        current_step = COLDSTART_STATE.get('current_step', 0)

        if current_step < initial_count:
            # Initial rating phase - move to next initial movie
            session_phase = 'initial_rating'
            message = "Movie skipped. Continue with initial rating."

            # Get next initial movie
            initial_movies = COLDSTART_STATE.get('current_movies', [])
            current_index = COLDSTART_STATE.get('current_movie_index', 0) + 1
            COLDSTART_STATE['current_movie_index'] = current_index

            if current_index < len(initial_movies):
                next_movie = initial_movies[current_index]
            else:
                next_movie = None
                message = "Initial movies completed."
        else:
            # Active learning phase
            session_phase = 'active_learning'
            message = "Movie skipped. Getting another recommendation..."
            next_movie = get_replacement_movie(
                COLDSTART_STATE['user_vector'],
                COLDSTART_STATE['rated_movies'],
                COLDSTART_STATE['skipped_movies']
            )

        response_data = {
            'message': message,
            'success': True,
            'next_movie': next_movie,
            'session_step': session_phase
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

        movie_id = int(data.get('movie_id'))
        if not movie_id:
            return JsonResponse({'error': 'Movie ID required'}, status=400)

        current_ratings = {d['movie_id']: d['rating'] for d in COLDSTART_STATE.get('rated_movies', [])}
        explanations = []
        cold_start = ColdStart(MOVIE_CACHE['V_matrix'], MOVIE_CACHE['R_matrix'])

        # Generate explanations for each possible rating (1-5)
        for hypothetical_rating in range(1, 6):
            explanation = generate_simple_explanation(
                current_ratings, movie_id, hypothetical_rating, cold_start
            )
            explanations.append({
                'rating': hypothetical_rating,
                'explanation': explanation
            })

        return JsonResponse({
            'success': True,
            'movie_id': movie_id,
            'explanations': explanations
        })

    except Exception as e:
        logger.error(f"Error getting rating explanation: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def generate_simple_explanation(current_ratings, movie_id, rating, cold_start):
    """Generate simplified explanation for rating impact"""
    try:
        simulated_ratings = current_ratings.copy()
        simulated_ratings[movie_id] = rating

        if current_ratings:
            current_user_vector = cold_start.update_user_vector(current_ratings)
        else:
            current_user_vector = np.zeros(MOVIE_CACHE['V_matrix'].shape[1])

        updated_user_vector = cold_start.update_user_vector(simulated_ratings)

        # Find next recommended movie
        next_movie = get_next_movie_recommendation(
            updated_user_vector,
            [{'movie_id': mid, 'rating': r} for mid, r in simulated_ratings.items()],
            COLDSTART_STATE.get('skipped_movies', [])
        )

        # Analyze feature changes
        feature_deltas = updated_user_vector - current_user_vector
        feature_changes = analyze_feature_changes(feature_deltas)

        # Calculate confidence
        confidence = min(0.95, max(0.3, rating / 5.0 * 0.7 + 0.3))

        return {
            'predicted_next_movie': {
                'title': next_movie['title'] if next_movie else 'No more movies',
                'confidence': confidence,
                'reason': f'Based on your rating of {rating} stars'
            },
            'feature_changes': feature_changes,
            'similarity_to_users': {
                'most_similar_user_type': get_user_type(rating),
                'similarity_score': confidence * 0.8
            }
        }

    except Exception as e:
        logger.error(f"Error in simple explanation: {str(e)}")
        return get_fallback_explanation(rating)


def analyze_feature_changes(feature_deltas):
    """Analyze which features changed most significantly"""
    top_indices = np.argsort(np.abs(feature_deltas))[::-1][:3]
    changes = []

    for idx in top_indices:
        if abs(feature_deltas[idx]) < 0.05:
            continue

        feature_key = f'Feature_{idx + 1}'
        feature_name = feature_dict.get(feature_key, f"Feature {idx + 1}")
        direction = "increase" if feature_deltas[idx] > 0 else "decrease"

        changes.append({
            'feature_name': feature_name,
            'change': direction,
            'magnitude': abs(feature_deltas[idx]),
            'description': f"Your preference for '{feature_name.lower()}' would {direction}"
        })

    if not changes:
        changes.append({
            'feature_name': 'Overall preferences',
            'change': 'adjust',
            'magnitude': 0.1,
            'description': 'Your taste profile would be refined'
        })

    return changes[:3]


def get_user_type(rating):
    """Simple user type based on rating"""
    if rating >= 4:
        return "Users who appreciate quality movies"
    elif rating <= 2:
        return "Users with selective taste"
    else:
        return "Users with balanced preferences"


def get_fallback_explanation(rating):
    """Fallback explanation when calculation fails"""
    return {
        'predicted_next_movie': {
            'title': 'Similar movies to your taste',
            'confidence': 0.5,
            'reason': f'Based on your {rating}-star rating'
        },
        'feature_changes': [{
            'feature_name': 'General preferences',
            'change': 'adjust',
            'magnitude': 0.2,
            'description': 'Your taste profile would be updated'
        }],
        'similarity_to_users': {
            'most_similar_user_type': 'General movie watchers',
            'similarity_score': 0.6
        }
    }


def reset_session():
    """Reset the cold start session"""
    global COLDSTART_STATE, MOVIE_CACHE

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

    initial_count = COLDSTART_STATE.get('initial_movies_count', 8)
    current_step = COLDSTART_STATE.get('current_step', 0)

    return {
        'session_active': True,
        'current_step': current_step,
        'initial_movies_count': initial_count,
        'total_ratings': COLDSTART_STATE.get('total_ratings', 0),
        'rated_count': len(COLDSTART_STATE.get('rated_movies', [])),
        'skipped_count': len(COLDSTART_STATE.get('skipped_movies', [])),
        'current_movies': COLDSTART_STATE.get('current_movies', []),
        'session_phase': 'initial_rating' if current_step < initial_count else 'active_learning'
    }


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
            'loaded': True,
            'selected_movies': None
        })

        if DEBUG:
            logger.info("Movie data loaded successfully")

    except Exception as e:
        logger.error(f"Error loading movie data: {str(e)}")
        raise


def get_initial_movie_selection(n):
    """Get initial movies for cold start using clustering"""
    if MOVIE_CACHE.get('selected_movies') is None:
        from project_4.Cold_start_recommendation.Clustering import run_true_hybrid_cold_start

        df_movies = MOVIE_CACHE['movies_df']
        df_ratings = MOVIE_CACHE['ratings_df']
        R_matrix = MOVIE_CACHE['R_matrix']
        V_matrix = MOVIE_CACHE['V_matrix']

        class MockModel:
            def __init__(self, V):
                self.V = V

        mock_model = MockModel(V_matrix)

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

    if 'sampled_movies' not in MOVIE_CACHE:
        MOVIE_CACHE['sampled_movies'] = random.sample(selected_movies, min(n, len(selected_movies)))
    return MOVIE_CACHE['sampled_movies']


def get_next_movie_recommendation(user_vector, rated_movies, skipped_movies):
    """Get next movie recommendation based on user vector"""
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
    V_candidates = V[candidate_indices, :]
    predicted_ratings = V_candidates @ user_vector
    predicted_ratings = np.clip(predicted_ratings, 1.0, 5.0)

    top_idx = np.argmax(predicted_ratings)
    best_movie_id = candidate_ids[top_idx]
    predicted_score = predicted_ratings[top_idx]

    movie_info = MOVIE_CACHE['movies_df'][MOVIE_CACHE['movies_df']['movieId'] == best_movie_id].iloc[0]

    return {
        'movie_id': int(best_movie_id),
        'title': movie_info['title'],
        'genres': movie_info['genres'],
        'predicted_rating': float(predicted_score)
    }


def get_replacement_movie(user_vector, rated_movies, skipped_movies):
    """Get replacement movie when user skips"""
    try:
        rated_ids = {m['movie_id'] for m in rated_movies}
        skipped_ids = {m['movie_id'] for m in skipped_movies}
        excluded_ids = rated_ids.union(skipped_ids)

        R = MOVIE_CACHE['R_matrix']
        V = MOVIE_CACHE['V_matrix']
        movie_ids_list = list(R.columns)

        candidate_ids = [i for i in R.columns if i not in excluded_ids]
        if not candidate_ids:
            return None

        if user_vector is None:
            import random
            selected_id = random.choice(candidate_ids)
            movie_info = MOVIE_CACHE['movies_df'][MOVIE_CACHE['movies_df']['movieId'] == selected_id].iloc[0]
            return {
                'movie_id': int(selected_id),
                'title': movie_info['title'],
                'genres': movie_info['genres']
            }

        candidate_indices = [movie_ids_list.index(i) for i in candidate_ids]
        V_candidates = V[candidate_indices, :]
        predicted_ratings = V_candidates @ user_vector
        predicted_ratings = np.clip(predicted_ratings, 1.0, 5.0)

        sorted_indices = np.argsort(predicted_ratings)[::-1]
        top_candidates = sorted_indices[:min(5, len(sorted_indices))]

        import random
        selected_idx = random.choice(top_candidates)
        best_movie_id = candidate_ids[selected_idx]
        predicted_score = predicted_ratings[selected_idx]

        movie_info = MOVIE_CACHE['movies_df'][MOVIE_CACHE['movies_df']['movieId'] == best_movie_id].iloc[0]

        return {
            'movie_id': int(best_movie_id),
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'predicted_rating': float(predicted_score)
        }

    except Exception as e:
        logger.error(f"Error getting replacement movie: {str(e)}")
        return None