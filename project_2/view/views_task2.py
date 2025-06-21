import ast
import json
import logging
import os

import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render

from pbl import settings
from project_2.active_learning.active_learning_loop import ActiveLearningLoop
from project_2.active_learning.utility_function import UtilityFunction
from project_2.view.view2_utils import create_model_from_pretrained_config, get_raw_text_for_sample
from project_2.view.views import DATA_STORAGE

DEBUG = True
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global state like the reference - resets with server restart
ACTIVE_LEARNING_STATE = {}

# Global cache for dataset
DATASET_CACHE = {
    'X': None,
    'y': None,
    'texts': None,
    'vectorizer': None,
    'loaded': False
}


def index(request):
    """Main view for Task 2: Active Learning - handles both GET and POST"""

    # Handle AJAX POST requests
    if request.method == 'POST':
        return handle_ajax_request(request)

    # GET request - render initial page
    baseline_accuracy = get_baseline_accuracy()
    has_pretrained_vectorizer = DATA_STORAGE.get('vectorizer') is not None
    has_pretrained_model = DATA_STORAGE.get('model_wrapper') is not None
    has_classifier_settings = DATA_STORAGE.get('classifier_settings') is not None
    has_complete_pretrained = has_pretrained_vectorizer and has_pretrained_model and has_classifier_settings

    context = {
        'has_pretrained': has_complete_pretrained,
        'has_pretrained_vectorizer': has_pretrained_vectorizer,
        'has_pretrained_model': has_pretrained_model,
        'has_classifier_settings': has_classifier_settings,
        'baseline_accuracy': baseline_accuracy,
        'classifier_settings': DATA_STORAGE.get('classifier_settings', None),
        'vectorizer_loaded': has_pretrained_vectorizer,
        'evaluation_results': DATA_STORAGE.get('evaluation_results', None),
        # Default configs
        'al_config': {'utility_function': 'lc', 'n_initial': 10, 'batch_size': 1},
        'termination': {'type': 'accuracy', 'target_accuracy': 0.85}
    }

    # Add current AL state if exists
    if ACTIVE_LEARNING_STATE.get('is_initialized'):
        context.update(get_current_al_context())

    return render(request, 'task2.html', context)


def handle_ajax_request(request):
    """Handle AJAX POST requests and return JSON responses"""
    try:
        data = json.loads(request.body)
        action = data.get('action')

        if DEBUG:
            logger.info(f"AJAX action: {action}")

        # Check for pre-trained model requirement
        has_complete_pretrained = (DATA_STORAGE.get('vectorizer') is not None and
                                   DATA_STORAGE.get('model_wrapper') is not None and
                                   DATA_STORAGE.get('classifier_settings') is not None)

        if action in ['initialize_al', 'label_sample', 'auto_query'] and not has_complete_pretrained:
            return JsonResponse({
                'error': "Active learning requires a complete pre-trained model from Task 1."
            }, status=400)

        if action == 'initialize_al':
            return initialize_active_learning(data)
        elif action == 'label_sample':
            return label_sample(data)
        elif action == 'auto_query':
            return auto_query(data)
        elif action == 'reset_al':
            return reset_active_learning()
        else:
            return JsonResponse({'error': f'Unknown action: {action}'}, status=400)

    except Exception as e:
        logger.error(f"Error handling AJAX request: {str(e)}")
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


def initialize_active_learning(data):
    """Initialize active learning - returns complete state"""
    global ACTIVE_LEARNING_STATE

    try:
        # Load data with pre-trained vectorizer
        X, y, raw_texts = ensure_data_loaded_with_pretrained()

        # Create model and utility function
        model = create_model_from_pretrained_config(logger)
        utility_function = create_utility_function(data)

        # Create and initialize AL loop
        al_loop = ActiveLearningLoop(X, y, utility_function, model, random_state=42)
        n_initial = int(data.get('n_initial', 10))
        al_loop.initialize_with_random_samples(n_initial)

        # Get next query
        next_query = None
        if len(al_loop.unlabeled_indices) > 0:
            query_idx, utility_scores = al_loop.get_next_query()
            raw_text = get_raw_text_for_sample(query_idx)
            next_query = {
                'sample_idx': int(query_idx),
                'text': raw_text,
                'utility_score': float(utility_scores.get(query_idx, 0))
            }

        # Store in global state
        ACTIVE_LEARNING_STATE = {
            'al_loop': al_loop,
            'is_initialized': True,
            'raw_texts': raw_texts,
            'config': data
        }

        # Return complete state
        response_data = {
            'message': f"Active learning initialized with {n_initial} samples",
            'success': True
        }
        response_data.update(get_current_al_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error initializing AL: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def label_sample(data):
    """Label a sample and return updated state"""
    global ACTIVE_LEARNING_STATE

    try:
        if not ACTIVE_LEARNING_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Active learning not initialized'}, status=400)

        al_loop = ACTIVE_LEARNING_STATE['al_loop']
        sample_idx = int(data.get('sample_idx'))
        use_oracle = data.get('use_oracle', False)

        if sample_idx not in al_loop.unlabeled_indices:
            return JsonResponse({'error': f'Sample {sample_idx} not available for labeling'}, status=400)

        if use_oracle:
            label = int(al_loop.y_train[sample_idx])
            message = f"Sample {sample_idx} auto-labeled as {'Positive' if label == 1 else 'Negative'} using oracle"
        else:
            label = int(data.get('label'))
            message = f"Sample {sample_idx} manually labeled as {'Positive' if label == 1 else 'Negative'}"

        # Query the sample
        result = al_loop.query_sample(sample_idx, label)
        message += f". New accuracy: {result['accuracy']:.3f}"

        # Get next query
        next_query = None
        if len(al_loop.unlabeled_indices) > 0:
            query_idx, utility_scores = al_loop.get_next_query()
            raw_text = get_raw_text_for_sample(query_idx)
            next_query = {
                'sample_idx': int(query_idx),
                'text': raw_text,
                'utility_score': float(utility_scores.get(query_idx, 0))
            }

        # Return complete updated state
        response_data = {
            'message': message,
            'success': True
        }
        response_data.update(get_current_al_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error labeling sample: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def auto_query(data):
    """Run automatic queries and return updated state"""
    global ACTIVE_LEARNING_STATE

    try:
        if not ACTIVE_LEARNING_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Active learning not initialized'}, status=400)

        al_loop = ACTIVE_LEARNING_STATE['al_loop']
        n_queries = int(data.get('n_queries', 10))

        # Run automatic loop
        results = al_loop.run_automatic_loop(n_queries)
        final_accuracy = results[-1]['accuracy'] if results else 0

        # Get next query
        next_query = None
        if len(al_loop.unlabeled_indices) > 0:
            query_idx, utility_scores = al_loop.get_next_query()
            raw_text = get_raw_text_for_sample(query_idx)
            next_query = {
                'sample_idx': int(query_idx),
                'text': raw_text,
                'utility_score': float(utility_scores.get(query_idx, 0))
            }

        # Return complete updated state
        response_data = {
            'message': f"Completed {len(results)} automatic queries. Final accuracy: {final_accuracy:.3f}",
            'success': True
        }
        response_data.update(get_current_al_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error during auto query: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def reset_active_learning():
    """Reset active learning state"""
    global ACTIVE_LEARNING_STATE

    try:
        ACTIVE_LEARNING_STATE = {}
        return JsonResponse({
            'message': 'Active learning session reset',
            'success': True
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_current_al_context():
    """Get current AL state for JSON response - like the reference"""
    if not ACTIVE_LEARNING_STATE.get('is_initialized'):
        return {}

    al_loop = ACTIVE_LEARNING_STATE['al_loop']

    # Current status
    al_status = {
        'is_initialized': True,
        'n_labeled': len(al_loop.labeled_indices),
        'n_unlabeled': len(al_loop.unlabeled_indices),
        'current_accuracy': al_loop.get_current_accuracy(),
        'model_trained': al_loop.model.is_trained,
        'n_queries_made': len(al_loop.accuracy_history) - 1 if al_loop.accuracy_history else 0
    }

    # Next query
    next_query = None
    if len(al_loop.unlabeled_indices) > 0:
        try:
            query_idx, utility_scores = al_loop.get_next_query()
            raw_text = get_raw_text_for_sample(query_idx)
            next_query = {
                'sample_idx': int(query_idx),
                'text': raw_text,
                'utility_score': float(utility_scores.get(query_idx, 0))
            }
        except Exception as e:
            logger.error(f"Error getting next query: {str(e)}")

    # Chart data
    accuracy_history = [float(acc) for acc in al_loop.accuracy_history] if al_loop.accuracy_history else []
    query_indices = list(range(len(accuracy_history)))

    return {
        'al_session': al_status,
        'al_status': al_status,
        'next_query': next_query,
        'accuracy_history': accuracy_history,
        'query_indices': query_indices,
        'baseline_accuracy': get_baseline_accuracy()
    }


def ensure_data_loaded_with_pretrained():
    """Load dataset with pre-trained vectorizer"""
    global DATASET_CACHE

    pretrained_vectorizer = DATA_STORAGE.get('vectorizer')
    if not pretrained_vectorizer:
        raise ValueError("No pre-trained vectorizer available")

    if DATASET_CACHE['loaded'] and DATASET_CACHE['vectorizer'] == pretrained_vectorizer:
        return DATASET_CACHE['X'], DATASET_CACHE['y'], DATASET_CACHE['texts']

    # Load data
    data_path_cleaned = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'cleaned_imdb_reviews.csv')
    data_path_raw = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'raw_imdb_reviews.csv')

    df_raw = pd.read_csv(data_path_raw)
    df_raw = df_raw[df_raw['review'].notna()]
    raw_texts = df_raw['review'].tolist()

    df_clean = pd.read_csv(data_path_cleaned)
    df_clean = df_clean[df_clean['review'].notna()]

    df_clean['review'] = df_clean['review'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    df_clean['review'] = df_clean['review'].apply(
        lambda tokens: " ".join(tokens) if isinstance(tokens, list) else str(tokens))

    cleaned_texts = df_clean['review'].tolist()
    labels = [1 if label == 'positive' else 0 for label in df_clean['sentiment'].tolist()]

    X = pretrained_vectorizer.transform(cleaned_texts)

    DATASET_CACHE.update({
        'X': X,
        'y': np.array(labels),
        'texts': raw_texts,
        'vectorizer': pretrained_vectorizer,
        'loaded': True
    })

    return X, labels, raw_texts


def create_utility_function(config):
    """Create utility function from config"""
    utility_name = config.get('utility_function', 'lc')

    if utility_name == 'hybrid':
        from project_2.active_learning.utility_function import UtilityFunction
        primary_util = UtilityFunction(config.get('primary_strategy', 'lc'))
        secondary_util = UtilityFunction(config.get('secondary_strategy', 'density'))

        class HybridUtilityFunction:
            def __init__(self, primary, secondary, alpha, combination_method):
                self.primary = primary
                self.secondary = secondary
                self.alpha = alpha
                self.combination_method = combination_method

            def apply(self, clf, X_pool):
                primary_scores = self.primary.apply(clf, X_pool)
                secondary_scores = self.secondary.apply(clf, X_pool)
                return self.primary.apply_hybrid(
                    self.combination_method, primary_scores, secondary_scores, self.alpha
                )

        return HybridUtilityFunction(
            primary_util, secondary_util,
            float(config.get('alpha', 0.5)),
            config.get('combination_method', 'sum')
        )
    else:
        from project_2.active_learning.utility_function import UtilityFunction
        return UtilityFunction(utility_name)


def get_baseline_accuracy():
    """Get baseline accuracy from pre-trained model"""
    model_wrapper = DATA_STORAGE.get('model_wrapper')
    if model_wrapper and hasattr(model_wrapper, 'metrics') and model_wrapper.metrics:
        return model_wrapper.metrics.get('accuracy')
    return None