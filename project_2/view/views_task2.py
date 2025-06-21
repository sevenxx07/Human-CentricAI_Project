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
        'al_config': {'utility_function': 'lc', 'n_initial': 10, 'batch_size': 1, 'diversity_method': 'top_k'},
        'termination': {'type': 'accuracy', 'target_accuracy': 0.85, 'max_queries': 100, 'budget_percent': 10}
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

        if action in ['initialize_al', 'label_sample', 'label_batch', 'auto_query'] and not has_complete_pretrained:
            return JsonResponse({
                'error': "Active learning requires a complete pre-trained model from Task 1."
            }, status=400)

        if action == 'initialize_al':
            return initialize_active_learning(data)
        elif action == 'label_sample':
            return label_sample(data)
        elif action == 'label_batch':
            return label_batch(data)
        elif action == 'auto_query':
            return auto_query(data)
        elif action == 'set_termination':
            return set_termination_conditions(data)
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

        # Get batch size and diversity method from config
        batch_size = int(data.get('batch_size', 1))
        diversity_method = data.get('diversity_method', 'top_k')

        # Get next query (single or batch)
        next_query = None
        if len(al_loop.unlabeled_indices) > 0 and not al_loop.termination_conditions['is_terminated']:
            if batch_size > 1:
                query_indices, utility_scores = al_loop.get_next_batch_query(batch_size, diversity_method)
                raw_texts_batch = [get_raw_text_for_sample(idx) for idx in query_indices]
                utility_scores_batch = [float(utility_scores.get(idx, 0)) for idx in query_indices]
                next_query = {
                    'sample_indices': [int(idx) for idx in query_indices],
                    'texts': raw_texts_batch,
                    'utility_scores': utility_scores_batch,
                    'batch_size': len(query_indices),
                    'diversity_method': diversity_method
                }
            else:
                query_idx, utility_scores = al_loop.get_next_query()
                raw_text = get_raw_text_for_sample(query_idx)
                next_query = {
                    'sample_idx': int(query_idx),
                    'text': raw_text,
                    'utility_score': float(utility_scores.get(query_idx, 0)),
                    'batch_size': 1
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
            'message': f"Active learning initialized with {n_initial} samples (batch size: {batch_size})",
            'success': True
        }
        response_data.update(get_current_al_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error initializing AL: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def set_termination_conditions(data):
    """Set termination conditions for active learning"""
    global ACTIVE_LEARNING_STATE

    try:
        if not ACTIVE_LEARNING_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Active learning not initialized'}, status=400)

        al_loop = ACTIVE_LEARNING_STATE['al_loop']
        termination_type = data.get('termination_type')

        termination_params = {}
        if 'target_accuracy' in data:
            termination_params['target_accuracy'] = float(data['target_accuracy'])
        if 'max_queries' in data:
            termination_params['max_queries'] = int(data['max_queries'])
        if 'budget_percent' in data:
            termination_params['budget_percent'] = float(data['budget_percent'])

        al_loop.set_termination_conditions(termination_type, **termination_params)

        # Check if already terminated
        is_terminated = al_loop.check_termination_conditions()

        message = f"Termination condition set: {termination_type}"
        if termination_type == 'accuracy':
            message += f" (target: {termination_params.get('target_accuracy', 0.85):.3f})"
        elif termination_type == 'queries':
            message += f" (max: {termination_params.get('max_queries', 100)})"
        elif termination_type == 'budget':
            message += f" (budget: {termination_params.get('budget_percent', 10):.1f}%)"

        if is_terminated:
            message += f" - Already met! {al_loop.termination_conditions['termination_reason']}"

        response_data = {
            'message': message,
            'success': True
        }
        response_data.update(get_current_al_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error setting termination conditions: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def label_sample(data):
    """Label a single sample and return updated state"""
    global ACTIVE_LEARNING_STATE

    try:
        if not ACTIVE_LEARNING_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Active learning not initialized'}, status=400)

        al_loop = ACTIVE_LEARNING_STATE['al_loop']

        # Check if already terminated
        if al_loop.termination_conditions['is_terminated']:
            return JsonResponse({
                'error': f"Active learning terminated: {al_loop.termination_conditions['termination_reason']}"
            }, status=400)

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

        # Check if terminated
        if result.get('is_terminated', False):
            message += f" - {result['termination_reason']}"

        # Get next query
        next_query = get_next_query_for_context(al_loop)

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


def label_batch(data):
    """Label a batch of samples and return updated state"""
    global ACTIVE_LEARNING_STATE

    try:
        if not ACTIVE_LEARNING_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Active learning not initialized'}, status=400)

        al_loop = ACTIVE_LEARNING_STATE['al_loop']

        # Check if already terminated
        if al_loop.termination_conditions['is_terminated']:
            return JsonResponse({
                'error': f"Active learning terminated: {al_loop.termination_conditions['termination_reason']}"
            }, status=400)

        sample_indices = [int(idx) for idx in data.get('sample_indices', [])]
        use_oracle = data.get('use_oracle', False)

        if not all(idx in al_loop.unlabeled_indices for idx in sample_indices):
            invalid_indices = [idx for idx in sample_indices if idx not in al_loop.unlabeled_indices]
            return JsonResponse({'error': f'Samples {invalid_indices} not available for labeling'}, status=400)

        if use_oracle:
            labels = [int(al_loop.y_train[idx]) for idx in sample_indices]
            message = f"Batch of {len(sample_indices)} samples auto-labeled using oracle"
        else:
            labels = [int(label) for label in data.get('labels', [])]
            if len(labels) != len(sample_indices):
                return JsonResponse({'error': 'Number of labels must match number of samples'}, status=400)
            message = f"Batch of {len(sample_indices)} samples manually labeled"

        # Query the batch
        result = al_loop.query_batch(sample_indices, labels)
        message += f". New accuracy: {result['accuracy']:.3f}"

        # Check if terminated
        if result.get('is_terminated', False):
            message += f" - {result['termination_reason']}"

        # Return complete updated state
        response_data = {
            'message': message,
            'success': True
        }
        response_data.update(get_current_al_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error labeling batch: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def auto_query(data):
    """Run automatic queries and return updated state"""
    global ACTIVE_LEARNING_STATE

    try:
        if not ACTIVE_LEARNING_STATE.get('is_initialized'):
            return JsonResponse({'error': 'Active learning not initialized'}, status=400)

        al_loop = ACTIVE_LEARNING_STATE['al_loop']

        # Check if already terminated
        if al_loop.termination_conditions['is_terminated']:
            return JsonResponse({
                'error': f"Active learning terminated: {al_loop.termination_conditions['termination_reason']}"
            }, status=400)

        n_queries = int(data.get('n_queries', 10))

        # Get batch configuration from stored config
        config = ACTIVE_LEARNING_STATE.get('config', {})
        batch_size = int(config.get('batch_size', 1))
        diversity_method = config.get('diversity_method', 'top_k')

        # Run automatic loop with batch support
        results = al_loop.run_automatic_loop(n_queries, batch_size=batch_size, diversity_method=diversity_method)

        if results:
            final_accuracy = results[-1]['accuracy']
            actual_queries = len(results)

            if batch_size > 1:
                total_samples = sum(result.get('batch_size', 1) for result in results)
                message = f"Completed {actual_queries} batches ({total_samples} total samples)"
            else:
                message = f"Completed {actual_queries} automatic queries"

            if actual_queries < n_queries:
                if al_loop.termination_conditions['is_terminated']:
                    message += f" (stopped early: {al_loop.termination_conditions['termination_reason']})"
                else:
                    message += " (no more unlabeled samples)"
            message += f". Final accuracy: {final_accuracy:.3f}"
        else:
            message = "No queries were performed (already terminated or no samples available)"

        # Return complete updated state
        response_data = {
            'message': message,
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


def get_next_query_for_context(al_loop):
    """Helper function to get next query based on current batch configuration"""
    config = ACTIVE_LEARNING_STATE.get('config', {})
    batch_size = int(config.get('batch_size', 1))
    diversity_method = config.get('diversity_method', 'top_k')

    if len(al_loop.unlabeled_indices) == 0 or al_loop.termination_conditions['is_terminated']:
        return None

    try:
        if batch_size > 1:
            query_indices, utility_scores = al_loop.get_next_batch_query(batch_size, diversity_method)
            raw_texts_batch = [get_raw_text_for_sample(idx) for idx in query_indices]
            utility_scores_batch = [float(utility_scores.get(idx, 0)) for idx in query_indices]
            return {
                'sample_indices': [int(idx) for idx in query_indices],
                'texts': raw_texts_batch,
                'utility_scores': utility_scores_batch,
                'batch_size': len(query_indices),
                'diversity_method': diversity_method
            }
        else:
            query_idx, utility_scores = al_loop.get_next_query()
            raw_text = get_raw_text_for_sample(query_idx)
            return {
                'sample_idx': int(query_idx),
                'text': raw_text,
                'utility_score': float(utility_scores.get(query_idx, 0)),
                'batch_size': 1
            }
    except Exception as e:
        logger.error(f"Error getting next query: {str(e)}")
        return None


def get_current_al_context():
    """Get current AL state for JSON response - like the reference"""
    if not ACTIVE_LEARNING_STATE.get('is_initialized'):
        return {}

    al_loop = ACTIVE_LEARNING_STATE['al_loop']

    # Current status
    al_status = al_loop.get_status()

    # Next query
    next_query = get_next_query_for_context(al_loop)

    # Chart data
    accuracy_history = [float(acc) for acc in al_loop.accuracy_history] if al_loop.accuracy_history else []
    query_indices = list(range(len(accuracy_history)))

    return {
        'al_session': al_status,
        'al_status': al_status,
        'next_query': next_query,
        'accuracy_history': accuracy_history,
        'query_indices': query_indices,
        'baseline_accuracy': get_baseline_accuracy(),
        'termination_conditions': al_loop.termination_conditions,
        'config': ACTIVE_LEARNING_STATE.get('config', {})
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