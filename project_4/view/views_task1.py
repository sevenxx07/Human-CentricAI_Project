import json
import logging
import os

from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings

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

        # TODO: Call clustering method to get initial 10 movies
        initial_movies = get_initial_movie_selection(n)

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

        # TODO: Update user vector based on rating
        updated_user_vector = update_user_vector(
            COLDSTART_STATE.get('user_vector'),
            movie_id,
            rating
        )
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

        # TODO: Get next movie recommendation
        next_movie = get_next_movie_recommendation(COLDSTART_STATE['user_vector'])
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

        # TODO: Get replacement movie (should not be affected by skip)
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

    # TODO: Load movie data from files
    # This should load movies.csv, ratings.csv, and pre-computed matrix factorization results
    try:
        # Load movie and rating data
        movies_path = os.path.join(settings.BASE_DIR, 'data', 'ml-latest-small', 'movies.csv')
        ratings_path = os.path.join(settings.BASE_DIR, 'data', 'ml-latest-small', 'ratings.csv')

        # TODO: Implement actual loading
        # movies_df = pd.read_csv(movies_path)
        # ratings_df = pd.read_csv(ratings_path)
        # R_matrix = load_rating_matrix()
        # V_matrix = load_item_features()

        MOVIE_CACHE.update({
            'movies_df': None,  # TODO: Load actual data
            'ratings_df': None,  # TODO: Load actual data
            'R_matrix': None,  # TODO: Load actual data
            'V_matrix': None,  # TODO: Load actual data
            'loaded': True
        })

        if DEBUG:
            logger.info("Movie data loaded successfully")

    except Exception as e:
        logger.error(f"Error loading movie data: {str(e)}")
        raise


# TODO: Implement these functions with actual clustering and recommendation logic

def get_initial_movie_selection(n=8):
    """
    TODO: Call clustering method to return initial list of n films
    This should use the clustering algorithm to select diverse movies
    for initial user preference learning
    """
    # Placeholder return - replace with actual clustering implementation
    initial_movies = [
        {
            'movie_id': i,
            'title': f'Sample Movie {i}',
            'genres': 'Action|Adventure',
            'year': 2020,
            'poster_url': None
        }
        for i in range(1, n)
    ]

    if DEBUG:
        logger.info(f"Generated {len(initial_movies)} initial movies for rating")

    return initial_movies


def update_user_vector(current_vector, movie_id, rating):
    """
    TODO: Update user's vector based on their rating input
    This should update the user's latent factor representation
    based on the movie they rated and the rating they gave
    """
    # Placeholder implementation
    if DEBUG:
        logger.info(f"Updating user vector with rating {rating} for movie {movie_id}")

    # TODO: Implement actual vector update logic
    updated_vector = current_vector  # Placeholder

    return updated_vector


def get_next_movie_recommendation(user_vector):
    """
    TODO: Query model to provide one new film prediction for user to rate
    This should use the updated user vector to find the most informative
    movie for the user to rate next
    """
    # Placeholder implementation
    if DEBUG:
        logger.info("Getting next movie recommendation")

    # TODO: Implement actual recommendation logic
    # This should find movies that would be most informative for the user to rate
    next_movie = {
        'movie_id': 999,
        'title': 'Recommended Movie',
        'genres': 'Drama|Romance',
        'year': 2021,
        'poster_url': None,
        'predicted_rating': 4.2
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