import ast
import logging
import os

import numpy as np
import pandas as pd
from django.contrib import messages
from django.shortcuts import render, redirect

from pbl import settings
from project_2.active_learning.active_learning_loop import ActiveLearningLoop
from project_2.active_learning.utility_function import UtilityFunction
from project_2.view.view2_utils import create_model_from_pretrained_config, get_raw_text_for_sample
from project_2.view.views import DATA_STORAGE  # Import the global storage

DEBUG = True
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global cache for dataset - load once and reuse
DATASET_CACHE = {
    'X': None,
    'y': None,
    'texts': None,
    'vectorizer': None,
    'loaded': False
}


class ActiveLearningConfig:
    """Configuration holder for active learning settings"""

    def __init__(self):
        self.utility_function = 'lc'
        self.primary_strategy = 'lc'
        self.secondary_strategy = 'density'
        self.alpha = 0.5
        self.combination_method = 'sum'
        self.n_initial = 10
        self.batch_size = 1


class TerminationConfig:
    """Configuration holder for termination conditions"""

    def __init__(self):
        self.type = 'accuracy'
        self.target_accuracy = 0.85
        self.max_queries = 100
        self.budget_percent = 10


def index(request):
    """Main view for Task 2: Active Learning"""

    # Initialize session storage if not exists
    if 'al_config' not in request.session:
        request.session['al_config'] = {}
    if 'al_session_data' not in request.session:
        request.session['al_session_data'] = {}
    if 'termination_config' not in request.session:
        request.session['termination_config'] = {}

    # Load configurations from session
    al_config = ActiveLearningConfig()
    for key, value in request.session['al_config'].items():
        setattr(al_config, key, value)

    termination = TerminationConfig()
    for key, value in request.session['termination_config'].items():
        setattr(termination, key, value)

    # Check for required pre-trained components
    baseline_accuracy = get_baseline_accuracy()
    has_pretrained_vectorizer = DATA_STORAGE.get('vectorizer') is not None
    has_pretrained_model = DATA_STORAGE.get('model_wrapper') is not None
    has_classifier_settings = DATA_STORAGE.get('classifier_settings') is not None

    # ALL components must be available for active learning
    has_complete_pretrained = has_pretrained_vectorizer and has_pretrained_model and has_classifier_settings

    if DEBUG:
        logger.info(f"Page load - Has vectorizer: {has_pretrained_vectorizer}, "
                    f"Has model: {has_pretrained_model}, Has settings: {has_classifier_settings}, "
                    f"Complete: {has_complete_pretrained}, Baseline: {baseline_accuracy}")

    # Initialize context
    context = {
        'al_config': al_config,
        'termination': termination,
        'al_session': None,
        'al_status': None,
        'next_query': None,
        'accuracy_history': [],
        'query_indices': [],
        'comparison_results': None,
        'scroll_to': None,
        'has_pretrained': has_complete_pretrained,  # Only true if ALL components available
        'has_pretrained_vectorizer': has_pretrained_vectorizer,
        'has_pretrained_model': has_pretrained_model,
        'has_classifier_settings': has_classifier_settings,
        'baseline_accuracy': baseline_accuracy,
        'classifier_settings': DATA_STORAGE.get('classifier_settings', None),
        'vectorizer_loaded': has_pretrained_vectorizer,
        'evaluation_results': DATA_STORAGE.get('evaluation_results', None)
    }

    if request.method == 'POST':
        action = request.POST.get('action')

        if DEBUG:
            logger.info(f"POST action: {action}")

        # Block all active learning actions if no complete pre-trained model
        if action in ['initialize_al', 'label_sample', 'auto_query',
                      'compare_strategies'] and not has_complete_pretrained:
            logger.warning(f"Blocked action '{action}' - no complete pre-trained model available")
            messages.error(request, "Active learning requires a complete pre-trained model from Task 1. "
                                    "Please train a model in Task 1 first.")
            return redirect('project2:task2')

        if action == 'initialize_al':
            return handle_initialize_al(request, al_config)
        elif action == 'label_sample':
            return handle_label_sample(request)
        elif action == 'auto_query':
            return handle_auto_query(request)
        elif action == 'set_termination':
            return handle_set_termination(request, termination)
        elif action == 'reset_al':
            return handle_reset_al(request)

    # Load lightweight session context if exists AND we have pre-trained model
    if (request.session.get('al_session_data') and
            request.session['al_session_data'].get('is_initialized') and
            has_complete_pretrained):
        try:
            context.update(load_lightweight_session_context(request))
        except Exception as e:
            logger.error(f"Error loading AL session context: {str(e)}")
            request.session.pop('al_session_data', None)
            messages.warning(request, "Active learning session was corrupted and has been reset")

    context['scroll_to'] = request.GET.get('scroll_to')

    return render(request, 'task2.html', context)


def handle_initialize_al(request, al_config):
    """Handle active learning initialization - requires pre-trained model"""
    try:
        # Strict check for pre-trained components
        if not DATA_STORAGE.get('vectorizer'):
            raise ValueError("No pre-trained vectorizer found. Please complete Task 1 first.")

        if not DATA_STORAGE.get('model_wrapper'):
            raise ValueError("No pre-trained model found. Please complete Task 1 first.")

        if not DATA_STORAGE.get('classifier_settings'):
            raise ValueError("No classifier settings found. Please complete Task 1 first.")

        # Update configuration from form
        al_config.utility_function = request.POST.get('utility_function', 'lc')
        al_config.n_initial = int(request.POST.get('n_initial', 10))
        al_config.batch_size = int(request.POST.get('batch_size', 1))

        if al_config.utility_function == 'hybrid':
            al_config.primary_strategy = request.POST.get('primary_strategy', 'lc')
            al_config.secondary_strategy = request.POST.get('secondary_strategy', 'density')
            al_config.alpha = float(request.POST.get('alpha', 0.5))
            al_config.combination_method = request.POST.get('combination_method', 'sum')

        # Save configuration to session
        request.session['al_config'] = al_config.__dict__

        if DEBUG:
            logger.info(f"Initializing AL with pre-trained model - utility: {al_config.utility_function}")

        # Load data from cache (loads only once) with REQUIRED pre-trained vectorizer
        X, y, raw_texts = ensure_data_loaded_with_pretrained()

        # Create NEW model instance (same type as pre-trained but fresh)
        model = create_model_from_pretrained_config(logger)
        utility_function = create_utility_function(al_config)

        # Create active learning loop
        al_loop = ActiveLearningLoop(X, y, utility_function, model, random_state=42)
        al_loop.initialize_with_random_samples(al_config.n_initial)

        # Get next query and cache it WITH RAW TEXT
        next_query_data = None
        if len(al_loop.unlabeled_indices) > 0:
            query_idx, utility_scores = al_loop.get_next_query()

            # Get raw text for display to user
            raw_text = get_raw_text_for_sample(query_idx)

            next_query_data = {
                'sample_idx': query_idx,
                'text': raw_text,  # This is the raw, human-readable text
                'utility_score': utility_scores.get(query_idx, 0)
            }

        # Store session data with cached next query
        session_data = {
            'labeled_indices': list(al_loop.labeled_indices),
            'unlabeled_indices': list(al_loop.unlabeled_indices),
            'accuracy_history': al_loop.accuracy_history,
            'cached_next_query': next_query_data,
            'is_initialized': True
        }
        request.session['al_session_data'] = session_data

        if DEBUG:
            logger.info(
                f"AL initialized with pre-trained components: {len(al_loop.labeled_indices)} labeled, "
                f"accuracy: {al_loop.get_current_accuracy():.3f}")
            if next_query_data:
                logger.info(f"Next query sample {query_idx}: '{next_query_data['text'][:100]}...'")

        messages.success(request,
                         f"Active learning initialized with {al_config.n_initial} samples using pre-trained model")

    except Exception as e:
        logger.error(f"Error initializing AL: {str(e)}")
        messages.error(request, f"Error initializing active learning: {str(e)}")

    return redirect('project2:task2')


def ensure_data_loaded_with_pretrained():
    """Ensure dataset is loaded with REQUIRED pre-trained vectorizer"""
    global DATASET_CACHE

    # Check for required pre-trained vectorizer
    pretrained_vectorizer = DATA_STORAGE.get('vectorizer')
    if not pretrained_vectorizer:
        raise ValueError("No pre-trained vectorizer available. Please complete Task 1 first.")

    if DATASET_CACHE['loaded'] and DATASET_CACHE['vectorizer'] == pretrained_vectorizer:
        return DATASET_CACHE['X'], DATASET_CACHE['y'], DATASET_CACHE['texts']

    if DEBUG:
        logger.info("Loading dataset with pre-trained vectorizer from Task 1...")

    # Use both cleaned and raw data files
    data_path_cleaned = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'cleaned_imdb_reviews.csv')
    data_path_raw = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'raw_imdb_reviews.csv')

    try:
        # Try to load raw data first for human display
        df_raw = pd.read_csv(data_path_raw)
        df_raw = df_raw[df_raw['review'].notna()]
        raw_texts = df_raw['review'].tolist()

        df_clean = pd.read_csv(data_path_cleaned)
        df_clean = df_clean[df_clean['review'].notna()]

        # Handle tokenized reviews from CLEANED data (for vectorization)
        df_clean['review'] = df_clean['review'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        df_clean['review'] = df_clean['review'].apply(
            lambda tokens: " ".join(tokens) if isinstance(tokens, list) else str(tokens))

        cleaned_texts = df_clean['review'].tolist()
        labels = [1 if label == 'positive' else 0 for label in df_clean['sentiment'].tolist()]

        # MUST use pre-trained vectorizer on CLEANED texts
        if DEBUG:
            logger.info("Using REQUIRED pre-trained vectorizer from Task 1 on cleaned texts")
        X = pretrained_vectorizer.transform(cleaned_texts)

        DATASET_CACHE.update({
            'X': X,
            'y': np.array(labels),
            'texts': raw_texts,
            'cleaned_texts': cleaned_texts,
            'vectorizer': pretrained_vectorizer,
            'loaded': True
        })

        if DEBUG:
            logger.info(
                f"Dataset loaded with pre-trained vectorizer: {X.shape} samples with {X.shape[1]} features")
            logger.info(f"Raw texts cached for human labeling interface: {len(raw_texts)} samples")

        return X, labels, raw_texts

    except Exception as e:
        logger.error(f"Error loading data with pre-trained vectorizer: {str(e)}")
        raise ValueError(f"Failed to load data with pre-trained components: {str(e)}")


def reconstruct_al_loop(request):
    """Reconstruct active learning loop from session data - requires pre-trained components"""
    session_data = request.session.get('al_session_data', {})
    al_config_data = request.session.get('al_config', {})

    if not session_data or not session_data.get('is_initialized'):
        raise ValueError("No active learning session found")

    # Strict check for pre-trained components
    if not DATA_STORAGE.get('vectorizer'):
        raise ValueError("No pre-trained vectorizer found. Cannot reconstruct AL session.")

    if not DATA_STORAGE.get('classifier_settings'):
        raise ValueError("No classifier settings found. Cannot reconstruct  AL session.")

    # Use cached data with REQUIRED pre-trained vectorizer
    X, y, texts = ensure_data_loaded_with_pretrained()

    # Create configuration object
    al_config = ActiveLearningConfig()
    for key, value in al_config_data.items():
        setattr(al_config, key, value)

    # Create model and utility function using pre-trained config
    model = create_model_from_pretrained_config(logger)
    utility_function = create_utility_function(al_config)

    # Create active learning loop
    al_loop = ActiveLearningLoop(X, y, utility_function, model, random_state=42)

    # Restore state
    al_loop.labeled_indices = set(session_data['labeled_indices'])
    al_loop.unlabeled_indices = set(session_data['unlabeled_indices'])
    al_loop.accuracy_history = session_data['accuracy_history']
    al_loop.is_initialized = True

    # Retrain model on current labeled data
    if al_loop.labeled_indices:
        al_loop._train_current_model()

    return al_loop


def get_baseline_accuracy() -> float | None:
    """Get baseline accuracy from pre-trained model if available"""
    model_wrapper = DATA_STORAGE.get('model_wrapper')
    if model_wrapper and hasattr(model_wrapper, 'metrics') and model_wrapper.metrics:
        accuracy = model_wrapper.metrics.get('accuracy')
        if accuracy is not None:
            return accuracy
    return None


def create_utility_function(config):
    """Create utility function based on configuration"""
    if config.utility_function == 'hybrid':
        # Create hybrid utility function
        primary_util = UtilityFunction(config.primary_strategy)
        secondary_util = UtilityFunction(config.secondary_strategy)

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
            primary_util, secondary_util, config.alpha, config.combination_method
        )
    else:
        return UtilityFunction(config.utility_function)


def load_lightweight_session_context(request):
    """Load session context without heavy computation - just display cached data"""
    session_data = request.session.get('al_session_data', {})

    if not session_data.get('is_initialized'):
        return {}

    # Create status from cached data without reconstructing the loop
    status = {
        'is_initialized': True,
        'n_labeled': len(session_data.get('labeled_indices', [])),
        'n_unlabeled': len(session_data.get('unlabeled_indices', [])),
        'current_accuracy': session_data.get('accuracy_history', [0])[-1] if session_data.get(
            'accuracy_history') else 0.0,
        'model_trained': True,
        'n_queries_made': len(session_data.get('accuracy_history', [])) - 1 if session_data.get(
            'accuracy_history') else 0  # Subtract initial accuracy
    }

    # Get next query from cached data if available
    next_query = session_data.get('cached_next_query')

    # Prepare chart data from cached history - ensure they're proper Python lists
    accuracy_history = session_data.get('accuracy_history', [])

    # Ensure accuracy_history is a list of floats
    if accuracy_history:
        accuracy_history = [float(acc) for acc in accuracy_history]

    # Create query indices starting from 0
    query_indices = list(range(len(accuracy_history)))

    if DEBUG:
        logger.info(f"Session context - Accuracy history: {len(accuracy_history)} points")
        logger.info(f"Query indices: {len(query_indices)} points")
        logger.info(f"First few accuracy values: {accuracy_history[:5] if accuracy_history else 'None'}")

    return {
        'al_session': status,
        'al_status': status,
        'next_query': next_query,
        'accuracy_history': accuracy_history,  # Return as list - template will handle conversion
        'query_indices': query_indices,         # Return as list - template will handle conversion
        'comparison_results': request.session.get('comparison_results')
    }


def handle_label_sample(request):
    """Handle manual sample labeling - FIXED oracle functionality"""
    try:
        sample_idx = int(request.POST.get('sample_idx'))
        use_oracle = request.POST.get('use_oracle') == 'true'

        if DEBUG:
            logger.info(f"Labeling sample {sample_idx}, use_oracle: {use_oracle}")

        al_loop = reconstruct_al_loop(request)         # Reconstruct active learning loop from session

        if sample_idx not in al_loop.unlabeled_indices:
            raise ValueError(f"Sample {sample_idx} is not available for labeling")

        if use_oracle:
            oracle_label = int(al_loop.y_train[sample_idx])  # Get true label from training set
            if DEBUG:
                logger.info(f"Using oracle: true label for sample {sample_idx} is {oracle_label}")
            result = al_loop.query_sample(sample_idx, oracle_label)
            messages.success(request,
                             f"Sample {sample_idx} auto-labeled as {'Positive' if oracle_label == 1 else 'Negative'} using oracle. New accuracy: {result['accuracy']:.3f}")
        else:
            # Manual labeling
            manual_label = int(request.POST.get('label'))
            if DEBUG:
                logger.info(f"Manual label for sample {sample_idx}: {manual_label}")
            result = al_loop.query_sample(sample_idx, manual_label)
            messages.success(request,
                             f"Sample {sample_idx} manually labeled as {'Positive' if manual_label == 1 else 'Negative'}. New accuracy: {result['accuracy']:.3f}")

        # Update session data with new next query
        update_session_data_with_next_query(request, al_loop)

        if DEBUG:
            logger.info(f"Sample {sample_idx} labeled, new accuracy: {result['accuracy']:.3f}")

    except Exception as e:
        logger.error(f"Error labeling sample: {str(e)}")
        messages.error(request, f"Error labeling sample: {str(e)}")

    return redirect(f"project2:task2")


def handle_auto_query(request):
    """Handle automatic queries using oracle"""
    try:
        n_queries = int(request.POST.get('n_queries', 10))

        # Reconstruct active learning loop from session
        al_loop = reconstruct_al_loop(request)

        # Run automatic queries
        results = al_loop.run_automatic_loop(n_queries)

        # Update session data with new next query
        update_session_data_with_next_query(request, al_loop)

        final_accuracy = results[-1]['accuracy'] if results else 0

        if DEBUG:
            logger.info(f"Completed {n_queries} auto queries, final accuracy: {final_accuracy:.3f}")

        messages.success(request, f"Completed {len(results)} automatic queries. Final accuracy: {final_accuracy:.3f}")

    except Exception as e:
        logger.error(f"Error during auto query: {str(e)}")
        messages.error(request, f"Error during automatic querying: {str(e)}")

    return redirect(f"project2:task2")


def handle_set_termination(request, termination):
    """Handle termination condition setting"""
    try:
        termination.type = request.POST.get('termination_type', 'accuracy')

        if termination.type == 'accuracy':
            termination.target_accuracy = float(request.POST.get('target_accuracy', 0.85))
        elif termination.type == 'queries':
            termination.max_queries = int(request.POST.get('max_queries', 100))
        elif termination.type == 'budget':
            termination.budget_percent = int(request.POST.get('budget_percent', 10))

        # Save to session
        request.session['termination_config'] = termination.__dict__

        messages.success(request, f"Termination condition set: {termination.type}")

    except Exception as e:
        logger.error(f"Error setting termination: {str(e)}")
        messages.error(request, f"Error setting termination: {str(e)}")

    return redirect(f"project2:task2")


def handle_reset_al(request):
    """Handle active learning reset"""
    # Clear session data
    request.session.pop('al_session_data', None)
    request.session.pop('comparison_results', None)

    if DEBUG:
        logger.info("Active learning session reset")

    messages.success(request, "Active learning session reset")
    return redirect('project2:task2')


def update_session_data_with_next_query(request, al_loop):
    """Update session data with current loop state and cache next query WITH RAW TEXT"""
    session_data = request.session.get('al_session_data', {})

    # Get next query and cache it WITH RAW TEXT
    next_query_data = None
    if len(al_loop.unlabeled_indices) > 0:
        query_idx, utility_scores = al_loop.get_next_query()

        # Get raw text for display to user
        try:
            raw_text = get_raw_text_for_sample(query_idx)
            next_query_data = {
                'sample_idx': int(query_idx),  # Convert numpy int64 to Python int
                'text': raw_text,  # This is the RAW, human-readable text
                'utility_score': float(utility_scores.get(query_idx, 0))  # Convert numpy float to Python float
            }

            if DEBUG:
                logger.info(f"Next query sample {query_idx}: '{raw_text[:100]}...'")

        except Exception as e:
            logger.error(f"Error getting raw text for sample {query_idx}: {str(e)}")
            # Fallback to cleaned text if raw text fails
            texts = DATASET_CACHE.get('cleaned_texts', DATASET_CACHE.get('texts', []))
            if query_idx < len(texts):
                next_query_data = {
                    'sample_idx': int(query_idx),  # Convert numpy int64 to Python int
                    'text': texts[query_idx],
                    'utility_score': float(utility_scores.get(query_idx, 0))  # Convert numpy float to Python float
                }

    # Convert numpy types to Python native types for JSON serialization
    session_data.update({
        'labeled_indices': [int(idx) for idx in al_loop.labeled_indices],  # Convert numpy ints
        'unlabeled_indices': [int(idx) for idx in al_loop.unlabeled_indices],  # Convert numpy ints
        'accuracy_history': [float(acc) for acc in al_loop.accuracy_history],  # Convert numpy floats
        'cached_next_query': next_query_data,
    })

    request.session['al_session_data'] = session_data
