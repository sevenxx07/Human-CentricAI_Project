import ast
import logging
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.shortcuts import render
from django.utils.timezone import now
from sklearn.model_selection import train_test_split

from project_2.ml_models.Representation import tfidf_representation, sbert_representation, glove_representation
from project_2.models import TextClassifier, TrainingSession

# Logging setup with debug info
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global variables for model persistence
model_global = None
vectorizer_global = None


def index(request):
    """Main view for Task 1: Supervised Learning"""
    logger.debug("=== Task 1 index() called ===")

    context = {
        'scroll_to': request.POST.get('scroll_to', 'step-1'),
        'selected_representation': 'tfidf',
        'classifier_settings': None,
        'training_session': None,
        'evaluation_results': None,
        'save_success': False,
        'save_error': None
    }

    # Add model choices to context
    context.update({
        'MODEL_TYPES': TextClassifier.MODEL_TYPES,
        'REPRESENTATIONS': TextClassifier.REPRESENTATIONS
    })

    # Get latest classifier if exists
    if TextClassifier.objects.exists():
        latest_classifier = TextClassifier.objects.latest('created_at')
        logger.debug(
            f"Found existing classifier: {latest_classifier.model_type} with {latest_classifier.representation_type}")
        context.update({
            'selected_representation': latest_classifier.representation_type,
            'classifier_settings': latest_classifier
        })
    else:
        # Create a dummy classifier_settings object for template compatibility
        context['classifier_settings'] = _create_dummy_classifier()

    if request.method == 'POST':
        action = request.POST.get('action')
        logger.info(f"POST action received: {action}")

        if action == 'select_model':
            return handle_model_selection(request, context)
        elif action == 'train_model':
            return handle_model_training(request, context)
        elif action == 'save_model':
            return handle_model_saving(request, context)

    return render(request, "task1.html", context)


def _create_dummy_classifier():
    """Create a dummy classifier for template compatibility"""

    class DummyClassifier:
        MODEL_TYPES = TextClassifier.MODEL_TYPES
        REPRESENTATIONS = TextClassifier.REPRESENTATIONS
        model_type = None
        representation_type = 'tfidf'

    return DummyClassifier()


def extract_hyperparameters(request, model_type):
    """Extract hyperparameters from request based on model type"""
    params = {}

    if model_type == 'logistic':
        params.update({
            'regularization_c': float(request.POST.get('log_C', 1.0)),
            'max_iter': int(request.POST.get('log_max_iter', 1000)),  # Changed from 'max_iter' to 'log_max_iter'
            'solver': request.POST.get('solver', 'lbfgs'),
            'penalty': request.POST.get('penalty', 'l2'),
        })
        logger.debug(
            f"Logistic regression params: C={params['regularization_c']}, max_iter={params['max_iter']}, solver={params['solver']}, penalty={params['penalty']}")
    elif model_type == 'svm':
        params.update({
            'regularization_c': float(request.POST.get('C', 1.0)),
            'kernel': request.POST.get('kernel', 'linear'),
            'gamma': request.POST.get('gamma', 'scale'),
            'max_iter': int(request.POST.get('max_iter', 1000))
        })
        logger.debug(
            f"SVM params: C={params['regularization_c']}, kernel={params['kernel']}, max_iter={params['max_iter']}")
    elif model_type == 'naive_bayes':
        params.update({
            'alpha': float(request.POST.get('alpha', 1.0)),
            'nb_variant': request.POST.get('nb_variant', 'gaussian'),
            'fit_prior': request.POST.get('fit_prior', 'true') == 'true',
        })
        logger.debug(
            f"Naive Bayes params: variant={params['nb_variant']}, alpha={params['alpha']}, fit_prior={params['fit_prior']}")

    return params


def handle_model_selection(request, context):
    """Handle model configuration"""
    logger.debug("=== handle_model_selection() ===")

    try:
        model_type = request.POST.get('model')
        representation_type = request.POST.get('representation', 'tfidf')

        logger.info(f"Configuring {model_type} with {representation_type} representation")
        logger.debug(f"POST data: {dict(request.POST)}")

        # Create or update classifier configuration
        classifier_data = {
            'name': f"{model_type}_{representation_type}_classifier",
            'model_type': model_type,
            'representation_type': representation_type,
        }

        # Add hyperparameters based on model type
        hyperparams = extract_hyperparameters(request, model_type)
        classifier_data.update(hyperparams)

        classifier_settings = TextClassifier.objects.create(**classifier_data)
        logger.info(f"Created classifier with ID: {classifier_settings.id}")

        context.update({
            'selected_representation': representation_type,
            'classifier_settings': classifier_settings,
            'scroll_to': 'step-2'
        })

        messages.success(request, f"{model_type.title()} model configured successfully!")

    except Exception as e:
        error_msg = f"Error configuring model: {str(e)}"
        logger.error(f"Model selection failed: {error_msg}")
        logger.debug(f"Exception type: {type(e).__name__}")
        context['error'] = error_msg
        messages.error(request, error_msg)

    return render(request, 'task1.html', context)


def load_data() -> Tuple[list, list]:
    """Load and preprocess the dataset"""
    data_path = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'cleaned_imdb_reviews.csv')
    logger.debug(f"Loading data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded CSV with {len(df)} rows")

    # Remove missing reviews
    initial_size = len(df)
    df = df[df['review'].notna()]
    final_size = len(df)

    if initial_size != final_size:
        logger.info(f"Removed {initial_size - final_size} rows with missing reviews")

    # Handle tokenized reviews if stored as string lists
    df['review'] = df['review'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    df['review'] = df['review'].apply(
        lambda tokens: " ".join(tokens) if isinstance(tokens, list) else str(tokens))

    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()

    logger.info(f"Final dataset: {len(texts)} texts with {len(set(labels))} unique labels")
    logger.debug(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    return texts, labels


def create_text_representation(texts: list, representation_type: str):
    """Create text representation"""
    logger.info(f"Creating {representation_type} representation for {len(texts)} texts")

    try:
        if representation_type == 'tfidf':
            X, vectorizer = tfidf_representation(texts)
        elif representation_type == 'sbert':
            X, vectorizer = sbert_representation(texts)
        elif representation_type == 'glove':
            X, vectorizer = glove_representation(texts)
        else:
            raise ValueError(f"Unknown representation type: {representation_type}")

        logger.info(f"Created representation with shape: {X.shape}")
        logger.debug(f"Representation type: {type(X)}")

        return X, vectorizer

    except Exception as e:
        logger.error(f"Failed to create {representation_type} representation: {str(e)}")
        raise


def prepare_training_data(classifier_settings):
    """Load and prepare training data"""
    logger.info("Loading dataset...")
    texts, labels = load_data()

    logger.info("Creating text representation...")
    X, vectorizer = create_text_representation(texts, classifier_settings.representation_type)
    y = np.array(labels)

    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, vectorizer


def train_model_with_strategy(model, classifier_settings, X_train, y_train):
    """Train model using appropriate strategy based on model type and data"""
    logger.info("Creating model instance...")

    # Handle Gaussian Naive Bayes with sparse matrices using batch training
    if (classifier_settings.model_type == 'naive_bayes' and
            classifier_settings.nb_variant == 'gaussian' and
            hasattr(X_train, 'toarray')):  # sparse matrix

        logger.info("Using batch training for Gaussian Naive Bayes with sparse matrix")
        _train_naive_bayes_in_batches(model, X_train, y_train)
    else:
        logger.info("Using standard training")
        model.train(X_train, y_train)

    return model


def _train_naive_bayes_in_batches(model, X_train, y_train):
    """Train Naive Bayes in batches to handle sparse matrices"""
    batch_size = 1000
    classes = np.unique(y_train)
    total_batches = (X_train.shape[0] + batch_size - 1) // batch_size
    logger.debug(f"Training in {total_batches} batches of size {batch_size}")

    for batch_idx, i in enumerate(range(0, X_train.shape[0], batch_size)):
        X_batch = X_train[i:i + batch_size].toarray()
        y_batch = y_train[i:i + batch_size]

        logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}: samples {i}-{i + len(X_batch)}")

        if i == 0:
            model.partial_fit(X_batch, y_batch, classes=classes)
        else:
            model.partial_fit(X_batch, y_batch)

    model.is_trained = True
    logger.info("Batch training completed")


def evaluate_trained_model(model, X_test, y_test):
    """Evaluate model on test set"""
    logger.info("Evaluating model on test set...")

    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Evaluation results: Accuracy={metrics['accuracy']:.4f}, "
                f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1_score']:.4f}")

    return metrics


def update_training_records(training_session, classifier_settings, metrics):
    """Update training session and classifier records"""
    # Update training session
    training_session.status = 'completed'
    training_session.metrics = metrics
    training_session.end_time = now()
    training_session.duration = training_session.end_time - training_session.start_time
    training_session.save()
    logger.info(f"Training session completed in {training_session.duration}")

    # Update classifier
    classifier_settings.is_trained = True
    classifier_settings.test_accuracy = metrics['accuracy']
    classifier_settings.save()


def handle_model_training(request, context):
    """Handle model training"""
    global model_global, vectorizer_global

    logger.debug("=== handle_model_training() ===")
    training_session = None
    context['scroll_to'] = 'step-3'

    try:
        # Get classifier settings
        classifier_settings = TextClassifier.objects.latest('created_at')
        if not classifier_settings:
            raise ValueError("No model configured. Please select a model first.")

        logger.info(f"Training {classifier_settings.model_type} with {classifier_settings.representation_type}")

        # Create training session
        training_session = TrainingSession.objects.create(status='running')
        context['training_session'] = training_session
        logger.info(f"Created training session: {training_session.id}")

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

        # Store for saving
        model_global = model
        vectorizer_global = vectorizer
        logger.debug("Stored model and vectorizer globally for saving")

        # Convert metrics to percentages for display

        context.update({
            'training_session': training_session,
            'classifier_settings': classifier_settings,
            'evaluation_results': display_metrics,
        })

        messages.success(request, "Model trained successfully!")

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Exception type: {type(e).__name__}")

        if training_session:
            training_session.status = 'failed'
            training_session.error_message = error_msg
            training_session.end_time = now()
            training_session.duration = training_session.end_time - training_session.start_time
            training_session.save()
            logger.info(f"Marked training session {training_session.id} as failed")

        context['error'] = error_msg
        messages.error(request, error_msg)

    return render(request, 'task1.html', context)


def save_model_and_vectorizer(model, vectorizer, representation_type):
    """Save model and vectorizer to disk"""
    # Create model directory
    model_dir = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'models')
    os.makedirs(model_dir, exist_ok=True)
    logger.debug(f"Model directory: {model_dir}")

    logger.info(f"Saving {representation_type} model and vectorizer")

    # Save model and vectorizer
    model.save_classifier(name_suffix=representation_type)

    vectorizer_path = os.path.join(model_dir, f"{representation_type}_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    logger.info(f"Model saved with suffix: {representation_type}")
    logger.info(f"Vectorizer saved to: {vectorizer_path}")


def handle_model_saving(request, context):
    """Handle model saving"""
    global model_global, vectorizer_global

    logger.debug("=== handle_model_saving() ===")

    try:
        if not model_global or not model_global.is_trained:
            raise ValueError("No trained model available to save")

        if not vectorizer_global:
            raise ValueError("No vectorizer available to save")

        # Get representation type for file naming
        representation_type = context.get('selected_representation', 'tfidf')

        save_model_and_vectorizer(model_global, vectorizer_global, representation_type)

        context['save_success'] = True
        messages.success(request, f"Model and {representation_type} vectorizer saved successfully!")

    except Exception as e:
        error_msg = f"Error saving model: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Exception type: {type(e).__name__}")
        context['save_error'] = error_msg
        messages.error(request, error_msg)

    context['scroll_to'] = 'step-3'
    return render(request, 'task1.html', context)
