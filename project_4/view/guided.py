import json
import logging
from django.http import JsonResponse
from django.shortcuts import render
from .session_utils import (
    MovieCache, ColdStartSession, initialize_session,
    process_rating_submission, process_movie_skip,
    generate_rating_explanation
)

# Configure logging
logger = logging.getLogger(__name__)

# Global state for guided version - separate from unguided
GUIDED_STATE = {}
GUIDED_MOVIE_CACHE = MovieCache()


# ==================== MAIN VIEW HANDLERS ====================

def index(request):
    """Main view for Guided Cold Start Active Learning"""
    if request.method == 'POST':
        return handle_ajax_request(request)

    context = {
        'session_active': GUIDED_STATE.get('session') is not None,
        'current_step': GUIDED_STATE.get('session').current_step if GUIDED_STATE.get('session') else 0,
        'total_ratings': GUIDED_STATE.get('session').total_ratings if GUIDED_STATE.get('session') else 0
    }

    if GUIDED_STATE.get('session'):
        context.update(GUIDED_STATE['session'].get_session_context())

    return render(request, 'project4_guided.html', context)


def handle_ajax_request(request):
    """Route AJAX requests to appropriate handlers"""
    try:
        data = json.loads(request.body)
        action = data.get('action')
        logger.info(f"AJAX action: {action}")

        handlers = {
            'initialize_session': initialize_guided_session,
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

def initialize_guided_session(data):
    """Initialize guided cold start session"""
    global GUIDED_STATE

    try:
        session = ColdStartSession(data.get('session_id', 'guided_session'))
        initial_movies = initialize_session(session, GUIDED_MOVIE_CACHE)

        GUIDED_STATE['session'] = session

        first_movie = initial_movies[0] if initial_movies else None

        response_data = {
            'message': 'Cold start session initialized successfully',
            'success': True,
            'movies': initial_movies,
            'session_step': 'initial_rating',
            'next_movie': first_movie
        }
        response_data.update(session.get_session_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error initializing guided session: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def reset_session(data=None):
    """Reset the guided session"""
    global GUIDED_STATE

    try:
        GUIDED_STATE = {}
        # Clear cached selections but keep expensive data loaded
        GUIDED_MOVIE_CACHE.selected_movies = None
        GUIDED_MOVIE_CACHE.sampled_movies = None

        return JsonResponse({
            'message': 'Cold start session reset successfully',
            'success': True
        })
    except Exception as e:
        logger.error(f"Error resetting guided session: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


# ==================== RATING HANDLERS ====================

def submit_rating(data):
    """Process user rating and update user vector"""
    try:
        session = GUIDED_STATE.get('session')
        if not session or not session.is_initialized:
            return JsonResponse({'error': 'Session not initialized'}, status=400)

        movie_id = data.get('movie_id')
        rating = float(data.get('rating'))

        if not movie_id or rating is None:
            return JsonResponse({'error': 'Movie ID and rating required'}, status=400)

        # Process rating and get next action
        next_movie, message, session_step = process_rating_submission(
            session, GUIDED_MOVIE_CACHE, movie_id, rating
        )

        response_data = {
            'message': message,
            'success': True,
            'session_step': session_step,
            'next_movie': next_movie
        }
        response_data.update(session.get_session_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error submitting rating: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def skip_movie(data):
    """Skip current movie without affecting user vector"""
    try:
        session = GUIDED_STATE.get('session')
        if not session or not session.is_initialized:
            return JsonResponse({'error': 'Session not initialized'}, status=400)

        movie_id = data.get('movie_id')
        if not movie_id:
            return JsonResponse({'error': 'Movie ID required'}, status=400)

        # Process skip and get next action
        next_movie, message, session_step = process_movie_skip(
            session, GUIDED_MOVIE_CACHE, movie_id
        )

        response_data = {
            'message': message,
            'success': True,
            'next_movie': next_movie,
            'session_step': session_step
        }
        response_data.update(session.get_session_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error skipping movie: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


# ==================== EXPLANATION SYSTEM ====================

def get_rating_explanation(data):
    """Generate explanations for different rating impacts using Cold_start functions"""
    try:
        session = GUIDED_STATE.get('session')
        if not session or not session.is_initialized:
            return JsonResponse({'error': 'Session not initialized'}, status=400)

        movie_id = int(data.get('movie_id'))
        if not movie_id:
            return JsonResponse({'error': 'Movie ID required'}, status=400)

        formatted_explanations = generate_rating_explanation(
            session, GUIDED_MOVIE_CACHE, movie_id
        )

        return JsonResponse({
            'success': True,
            'movie_id': movie_id,
            'explanations': formatted_explanations
        })

    except Exception as e:
        logger.error(f"Error getting rating explanation: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)