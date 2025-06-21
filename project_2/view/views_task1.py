import ast
import json
import logging
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.utils.timezone import now
from sklearn.model_selection import train_test_split

from project_2.ml_models.Representation import tfidf_representation, sbert_representation, glove_representation
from project_2.models import TextClassifier, TrainingSession
from project_2.view.views import DATA_STORAGE

DEBUG = True
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global state like the reference
TASK1_STATE = {}


def index(request):
    """Main view for Task 1: Supervised Learning - handles both GET and POST"""

    # Handle AJAX POST requests
    if request.method == 'POST':
        return handle_ajax_request(request)

    # GET request - render initial page
    context = {
        'MODEL_TYPES': TextClassifier.MODEL_TYPES,
        'REPRESENTATIONS': TextClassifier.REPRESENTATIONS,
        'selected_representation': 'tfidf',
        'classifier_settings': _create_dummy_classifier(),
        'training_session': None,
        'evaluation_results': None,
        'save_success': False
    }

    # Get latest classifier if exists
    if TextClassifier.objects.exists():
        latest_classifier = TextClassifier.objects.latest('created_at')
        context.update({
            'selected_representation': latest_classifier.representation_type,
            'classifier_settings': latest_classifier
        })

    # Add current state if exists
    if TASK1_STATE.get('configured'):
        context.update(get_current_task1_context_for_template())

    return render(request, "task1.html", context)


def handle_ajax_request(request):
    """Handle AJAX POST requests and return JSON responses"""
    try:
        data = json.loads(request.body)
        action = data.get('action')

        if DEBUG:
            logger.info(f"AJAX action: {action}")

        if action == 'select_model':
            return configure_model(data)
        elif action == 'train_model':
            return train_model(data)
        elif action == 'save_model':
            return save_model()
        else:
            return JsonResponse({'error': f'Unknown action: {action}'}, status=400)

    except Exception as e:
        logger.error(f"Error handling AJAX request: {str(e)}")
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


def configure_model(data):
    """Handle model configuration"""
    global TASK1_STATE

    try:
        model_type = data.get('model')
        representation_type = data.get('representation', 'tfidf')

        # Extract hyperparameters
        hyperparams = extract_hyperparameters(data, model_type)

        # Create classifier configuration
        classifier_data = {
            'name': f"{model_type}_{representation_type}_classifier",
            'model_type': model_type,
            'representation_type': representation_type,
            **hyperparams
        }

        classifier_settings = TextClassifier.objects.create(**classifier_data)

        # Store in global state
        TASK1_STATE = {
            'configured': True,
            'classifier_settings': classifier_settings,
            'model_type': model_type,
            'representation_type': representation_type
        }

        # Return complete state
        response_data = {
            'message': f"{model_type.title()} model configured successfully!",
            'success': True
        }
        response_data.update(get_current_task1_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error configuring model: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def train_model(data):
    """Handle model training"""
    global TASK1_STATE

    try:
        if not TASK1_STATE.get('configured'):
            return JsonResponse({'error': 'No model configured'}, status=400)

        classifier_settings = TASK1_STATE['classifier_settings']

        # Create training session
        training_session = TrainingSession.objects.create(status='running')

        # Prepare data
        X_train, X_test, y_train, y_test, vectorizer = prepare_training_data(classifier_settings)

        # Create and train model
        model = classifier_settings.create_model_instance()
        model = train_model_with_strategy(model, classifier_settings, X_train, y_train)

        # Evaluate model
        metrics = evaluate_trained_model(model, X_test, y_test)
        display_metrics = {k: v * 100 for k, v in metrics.items()}

        # Update records
        update_training_records(training_session, classifier_settings, display_metrics)

        # Store model and vectorizer for saving, and in DATA_STORAGE for Task 2
        TASK1_STATE.update({
            'trained': True,
            'model': model,
            'vectorizer': vectorizer,
            'training_session': training_session,
            'evaluation_results': display_metrics
        })

        # Store in DATA_STORAGE for Task 2 to use
        DATA_STORAGE['model_wrapper'] = model
        DATA_STORAGE['vectorizer'] = vectorizer
        DATA_STORAGE['classifier_settings'] = classifier_settings
        DATA_STORAGE['evaluation_results'] = display_metrics

        # Return complete state
        response_data = {
            'message': 'Model trained successfully!',
            'success': True
        }
        response_data.update(get_current_task1_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def save_model():
    """Handle model saving"""
    global TASK1_STATE

    try:
        if not TASK1_STATE.get('trained'):
            return JsonResponse({'error': 'No trained model available to save'}, status=400)

        model = TASK1_STATE['model']
        vectorizer = TASK1_STATE['vectorizer']
        representation_type = TASK1_STATE['representation_type']

        # Save model and vectorizer
        save_model_and_vectorizer(model, vectorizer, representation_type)

        TASK1_STATE['saved'] = True

        response_data = {
            'message': f"Model and {representation_type} vectorizer saved successfully!",
            'success': True
        }
        response_data.update(get_current_task1_context())

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def serialize_classifier_settings(classifier_settings):
    """Convert Django model instance to JSON-serializable dictionary"""
    if not classifier_settings:
        return None

    return {
        'id': classifier_settings.id,
        'name': classifier_settings.name,
        'model_type': classifier_settings.model_type,
        'representation_type': classifier_settings.representation_type,
        'is_trained': classifier_settings.is_trained,
        'test_accuracy': classifier_settings.test_accuracy,
        'regularization_c': classifier_settings.regularization_c,
        'max_iter': classifier_settings.max_iter,
        'solver': classifier_settings.solver,
        'penalty': classifier_settings.penalty,
        'nb_variant': classifier_settings.nb_variant,
        'fit_prior': classifier_settings.fit_prior,
        'alpha': classifier_settings.alpha,
        'kernel': classifier_settings.kernel,
        'gamma': classifier_settings.gamma,
        'hyperparameters': classifier_settings.get_hyperparameters()
    }


def serialize_training_session(training_session):
    """Convert Django training session to JSON-serializable dictionary"""
    if not training_session:
        return None

    return {
        'id': training_session.id,
        'status': training_session.status,
        'start_time': training_session.start_time.isoformat() if training_session.start_time else None,
        'end_time': training_session.end_time.isoformat() if training_session.end_time else None,
        'duration': str(training_session.duration) if training_session.duration else None,
        'final_accuracy': training_session.final_accuracy,
        'final_precision': training_session.final_precision,
        'final_recall': training_session.final_recall,
        'final_f1': training_session.final_f1
    }


def get_current_task1_context():
    """Get current Task 1 state for JSON response - serialized for AJAX"""
    if not TASK1_STATE.get('configured'):
        return {}

    context = {
        'configured': True,
        'classifier_settings': serialize_classifier_settings(TASK1_STATE['classifier_settings']),
        'selected_representation': TASK1_STATE['representation_type']
    }

    if TASK1_STATE.get('trained'):
        context.update({
            'trained': True,
            'training_session': serialize_training_session(TASK1_STATE['training_session']),
            'evaluation_results': TASK1_STATE['evaluation_results']
        })

    if TASK1_STATE.get('saved'):
        context['save_success'] = True

    return context


def get_current_task1_context_for_template():
    """Get current Task 1 state for template rendering - with Django model instances"""
    if not TASK1_STATE.get('configured'):
        return {}

    context = {
        'configured': True,
        'classifier_settings': TASK1_STATE['classifier_settings'],  # Keep as Django model for template
        'selected_representation': TASK1_STATE['representation_type']
    }

    if TASK1_STATE.get('trained'):
        context.update({
            'trained': True,
            'training_session': TASK1_STATE['training_session'],
            'evaluation_results': TASK1_STATE['evaluation_results']
        })

    if TASK1_STATE.get('saved'):
        context['save_success'] = True

    return context


def _create_dummy_classifier():
    """Create a dummy classifier for template compatibility"""

    class DummyClassifier:
        MODEL_TYPES = TextClassifier.MODEL_TYPES
        REPRESENTATIONS = TextClassifier.REPRESENTATIONS
        model_type = None
        representation_type = 'tfidf'
        is_trained = False

    return DummyClassifier()


def extract_hyperparameters(data, model_type):
    """Extract hyperparameters from data based on model type"""
    params = {}

    if model_type == 'logistic':
        params.update({
            'regularization_c': float(data.get('log_C', 1.0)),
            'max_iter': int(data.get('log_max_iter', 1000)),
            'solver': data.get('solver', 'lbfgs'),
            'penalty': data.get('penalty', 'l2'),
        })
    elif model_type == 'svm':
        params.update({
            'regularization_c': float(data.get('C', 1.0)),
            'kernel': data.get('kernel', 'linear'),
            'gamma': data.get('gamma', 'scale'),
            'max_iter': int(data.get('max_iter', 1000))
        })
    elif model_type == 'naive_bayes':
        params.update({
            'alpha': float(data.get('alpha', 1.0)),
            'nb_variant': data.get('nb_variant', 'gaussian'),
            'fit_prior': data.get('fit_prior', True),
        })

    return params


def load_data() -> Tuple[list, list]:
    """Load and preprocess the dataset"""
    data_path = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'cleaned_imdb_reviews.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df[df['review'].notna()]

    # Handle tokenized reviews
    df['review'] = df['review'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    df['review'] = df['review'].apply(
        lambda tokens: " ".join(tokens) if isinstance(tokens, list) else str(tokens))

    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()

    return texts, labels


def create_text_representation(texts: list, representation_type: str):
    """Create text representation"""
    if representation_type == 'tfidf':
        X, vectorizer = tfidf_representation(texts)
    elif representation_type == 'sbert':
        X, vectorizer = sbert_representation(texts)
    elif representation_type == 'glove':
        X, vectorizer = glove_representation(texts)
    else:
        raise ValueError(f"Unknown representation type: {representation_type}")

    return X, vectorizer


def prepare_training_data(classifier_settings):
    """Load and prepare training data"""
    texts, labels = load_data()
    X, vectorizer = create_text_representation(texts, classifier_settings.representation_type)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, vectorizer


def train_model_with_strategy(model, classifier_settings, X_train, y_train):
    """Train model using appropriate strategy"""
    if (classifier_settings.model_type == 'naive_bayes' and
            classifier_settings.nb_variant == 'gaussian' and
            hasattr(X_train, 'toarray')):
        _train_naive_bayes_in_batches(model, X_train, y_train)
    else:
        model.train(X_train, y_train)

    return model


def _train_naive_bayes_in_batches(model, X_train, y_train):
    """Train Naive Bayes in batches to handle sparse matrices"""
    batch_size = 1000
    classes = np.unique(y_train)

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i + batch_size].toarray()
        y_batch = y_train[i:i + batch_size]

        if i == 0:
            model.partial_fit(X_batch, y_batch, classes=classes)
        else:
            model.partial_fit(X_batch, y_batch)

    model.is_trained = True


def evaluate_trained_model(model, X_test, y_test):
    """Evaluate model on test set"""
    return model.evaluate(X_test, y_test)


def update_training_records(training_session, classifier_settings, metrics):
    """Update training session and classifier records"""
    training_session.status = 'completed'
    training_session.end_time = now()
    training_session.duration = training_session.end_time - training_session.start_time
    training_session.save()

    classifier_settings.is_trained = True
    classifier_settings.test_accuracy = metrics['accuracy']
    classifier_settings.save()


def save_model_and_vectorizer(model, vectorizer, representation_type):
    """Save model and vectorizer to disk"""
    model_dir = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'models')
    os.makedirs(model_dir, exist_ok=True)

    model.save_classifier(name_suffix=representation_type)

    vectorizer_path = os.path.join(model_dir, f"{representation_type}_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)