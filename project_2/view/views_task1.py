import ast
import os
import logging
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import pickle
import numpy as np
import scipy.sparse
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from django.conf import settings
from django.shortcuts import render
from django.contrib import messages
from django.utils.timezone import now
from django.core.exceptions import ValidationError

from project_2.ml_models.Representation import tfidf_representation, sbert_representation, glove_representation
from project_2.models import TextClassifier, TrainingSession

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEBUG = True  # Set to False in production
model_global = None  # Global variable to hold the model instance
vectorizer_global = None  # Global variable to hold the vectorizer instance


class TrainingError(Exception):
    """Custom exception for training-related errors"""
    pass


def index(request):
    """Main view for the text classification interface"""
    context = {
        'error': None,
        'scroll_to': request.POST.get('scroll_to', 'step-1'),
        'selected_representation': 'tfidf',  # Default value
        'classifier_settings': None,
        'training_session': None,
    }

    if DEBUG:
        logger.info("=== DEBUG: index() called ===")
        logger.debug(f"Initial context: {context}")

    # Always add these to context, regardless of whether classifier exists
    from project_2.models import TextClassifier
    context.update({
        'MODEL_TYPES': TextClassifier.MODEL_TYPES,
        'REPRESENTATIONS': TextClassifier.REPRESENTATIONS
    })

    # Get the most recent classifier if exists
    if TextClassifier.objects.exists():
        latest_classifier = TextClassifier.objects.latest('created_at')
        context.update({
            'selected_representation': latest_classifier.representation_type,
            'classifier_settings': latest_classifier
        })
    else:
        # Create a dummy classifier_settings object for template compatibility
        context['classifier_settings'] = _create_dummy_classifier()

    if request.method == 'POST':
        action = request.POST.get('action')
        if DEBUG:
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


def update_classifier_data(model_type: str, request, classifier_data: Dict[str, Any]) -> None:
    """Update classifier data based on the model type and request parameters."""

    try:
        if model_type == 'logistic':
            classifier_data.update({
                'regularization_c': float(request.POST.get('log_C', 1.0)),
                'max_iter': int(request.POST.get('max_iter', 1000)),
                'solver': request.POST.get('solver', 'lbfgs'),
                'penalty': request.POST.get('penalty', 'l2'),
            })
        elif model_type == 'svm':
            classifier_data.update({
                'regularization_c': float(request.POST.get('C', 1.0)),
                'kernel': request.POST.get('kernel', 'linear'),
                'gamma': request.POST.get('gamma', 'scale'),
            })
        elif model_type == 'naive_bayes':
            classifier_data.update({
                'alpha': float(request.POST.get('alpha', 1.0)),
                'nb_variant': request.POST.get('nb_variant', 'gaussian'),
                'fit_prior': request.POST.get('fit_prior', 'true') == 'true',
            })
        else:
            raise ValidationError(f"Unknown model type: {model_type}")

    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid hyperparameter values: {str(e)}")


def handle_model_selection(request, context):
    """Handle model type and hyperparameter selection"""

    try:
        if DEBUG:
            logger.info("=== DEBUG: handle_model_selection() ===")
            logger.debug(f"Incoming POST data: {dict(request.POST)}")

        model_type = request.POST.get('model')
        representation_type = request.POST.get('representation', 'tfidf')

        if not model_type:
            raise ValidationError("Model type is required")

        # Create or update classifier configuration
        classifier_data = {
            'name': f"{model_type}_{representation_type}_classifier",
            'model_type': model_type,
            'representation_type': representation_type,
        }

        update_classifier_data(model_type, request, classifier_data)
        classifier_settings = TextClassifier.objects.create(**classifier_data)

        context.update({
            'model_selected': True,
            'selected_representation': representation_type,
            'classifier_settings': classifier_settings,
            'scroll_to': 'step-2'
        })

        messages.success(request, f"{model_type.title()} model configured successfully!")
        if DEBUG:
            logger.info("Model configured successfully")

    except (ValidationError, Exception) as e:
        error_msg = f"Error configuring model: {str(e)}"
        context['error'] = error_msg
        messages.error(request, error_msg)
        logger.error(f"Error in model selection: {error_msg}")

    return render(request, 'task1.html', context)


def load_data(data_path: str) -> Tuple[list, list]:
    """Load and preprocess the dataset"""

    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)

        if df.empty:
            raise ValueError("Dataset is empty")

        # Check required columns
        required_columns = ['review', 'sentiment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Clean and preprocess data
        initial_size = len(df)
        df = df[df['review'].notna()]
        final_size = len(df)

        if DEBUG:
            logger.info(f"Removed {initial_size - final_size} rows with missing reviews")

        # Handle tokenized reviews (if they're stored as strings of lists)
        df['review'] = df['review'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        df['review'] = df['review'].apply(lambda tokens: " ".join(tokens) if isinstance(tokens, list) else str(tokens))

        texts = df['review'].tolist()
        labels = df['sentiment'].tolist()

        if DEBUG:
            logger.info(f"Loaded {len(texts)} texts with {len(set(labels))} unique labels")

        return texts, labels

    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {str(e)}")
        raise TrainingError(f"Failed to load data: {str(e)}")


def create_text_representation(texts: list, representation_type: str):
    """Create text representation based on the specified type"""

    try:
        if DEBUG:
            logger.info(f"Creating {representation_type} representation for {len(texts)} texts")

        if representation_type == 'tfidf':
            X, vectorizer = tfidf_representation(texts)
        elif representation_type == 'sbert':
            X, vectorizer = sbert_representation(texts)
        elif representation_type == 'glove':
            X, vectorizer = glove_representation(texts)
        else:
            raise ValueError(f"Unknown representation type: {representation_type}")

        if DEBUG:
            logger.info(f"Created representation with shape: {X.shape}")
            logger.info(f"Representation is sparse: {scipy.sparse.issparse(X)}")

        return X, vectorizer

    except Exception as e:
        logger.error(f"Error creating {representation_type} representation: {str(e)}")
        raise TrainingError(f"Failed to create text representation: {str(e)}")


def train_naive_bayes_in_batches(model, X_sparse, y, batch_size: int = 1000, training_session=None):
    """Train Naive Bayes incrementally to avoid memory issues"""

    try:
        if DEBUG:
            logger.info(f"Starting batch training for Naive Bayes with batch_size={batch_size}")

        # Validate inputs
        if X_sparse.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples")

        # Get unique classes for partial_fit
        classes = np.unique(y)
        if DEBUG:
            logger.info(f"Found {len(classes)} unique classes: {classes}")

        # Shuffle data
        X_sparse, y = shuffle(X_sparse, y, random_state=42)
        total_batches = (X_sparse.shape[0] + batch_size - 1) // batch_size

        # Train in batches
        for batch_idx, i in enumerate(range(0, X_sparse.shape[0], batch_size)):
            end_idx = min(i + batch_size, X_sparse.shape[0])
            X_batch = X_sparse[i:end_idx].astype('float32').toarray()  # Convert small batch to dense
            y_batch = y[i:end_idx]

            if DEBUG:
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches}, samples {i}-{end_idx}")

            # Update training session status
            if training_session:
                training_session.status = f'training batch {batch_idx + 1}/{total_batches}'
                training_session.save()

            if batch_idx == 0:
                # First batch needs classes parameter
                model.partial_fit(X_batch, y_batch, classes=classes)
            else:
                model.partial_fit(X_batch, y_batch)

        model.is_trained = True
        if DEBUG:
            logger.info("Batch training completed successfully")

        return model

    except Exception as e:
        logger.error(f"Error in batch training: {str(e)}")
        raise TrainingError(f"Batch training failed: {str(e)}")


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Evaluate the model and return metrics"""

    try:
        if DEBUG:
            logger.info(f"Evaluating model on {len(y_test)} test samples")

        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred) * 100,
            'precision': precision_score(y_test, y_pred, average='weighted') * 100,
            'recall': recall_score(y_test, y_pred, average='weighted') * 100,
            'f1': f1_score(y_test, y_pred, average='weighted') * 100
        }

        if DEBUG:
            logger.info("Evaluation metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.2f}%")

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise TrainingError(f"Model evaluation failed: {str(e)}")


def handle_model_training(request, context, data_path: Optional[str] = None):
    """Handle model training process with improved error handling and logging"""

    global model_global, vectorizer_global

    if DEBUG:
        logger.info("=== DEBUG: handle_model_training() ===")

    training_session = None
    context.update({'scroll_to': 'step-3'})

    # Set default data path
    if data_path is None:
        data_path = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'cleaned_imdb_reviews.csv')

    try:
        # 1. Get classifier settings
        classifier_settings = _get_latest_classifier_settings(context)

        # 2. Create training session
        training_session = TrainingSession.objects.create(status='initializing')
        context.update({'training_session': training_session})

        if DEBUG:
            logger.info(f"Training session created: {training_session.id}")
            logger.info(f"Starting training for model: {classifier_settings.model_type}")

        # 3. Load and prepare data
        training_session.status = 'loading data'
        logger.info(f"Training session updated: {training_session.status}")
        training_session.save()

        texts, labels = load_data(data_path)

        training_session.status = 'data loaded'
        logger.info(f"Training session updated: {training_session.status}")
        training_session.save()

        # 4. Create text representation
        training_session.status = 'creating representation'
        logger.info(f"Training session updated: {training_session.status}")
        training_session.save()

        X, vectorizer = create_text_representation(texts, classifier_settings.representation_type)
        y = np.array(labels)

        training_session.status = 'data vectorized'
        logger.info(f"Training session updated: {training_session.status}")
        training_session.save()

        # 5. Split data
        training_session.status = 'splitting data'
        logger.info(f"Training session updated: {training_session.status}")
        training_session.save()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if DEBUG:
            logger.info(f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        training_session.status = 'data split'
        logger.info(f"Training session updated: {training_session.status}")
        training_session.save()

        # 6. Create and train model
        training_session.status = 'training model'
        logger.info(f"Training session updated: {training_session.status}")
        training_session.save()

        model = classifier_settings.create_model_instance()

        # Special handling for Gaussian Naive Bayes with sparse matrices
        if _should_use_batch_training(classifier_settings, X_train):
            if DEBUG:
                logger.info("Using batch training for Gaussian Naive Bayes with sparse matrix")

            batch_size = min(1000, max(100, X_train.shape[0] // 20))  # Adaptive batch size
            model = train_naive_bayes_in_batches(
                model, X_train, y_train,
                batch_size=batch_size,
                training_session=training_session
            )
        else:
            if DEBUG:
                logger.info("Using normal training")
            model.train(X_train, y_train)

        training_session.status = 'model trained'
        training_session.save()

        # 7. Evaluate model
        training_session.status = 'evaluating model'
        training_session.save()

        metrics = evaluate_model(model, X_test, y_test)

        # 8. Update training session and classifier
        _update_training_completion(training_session, classifier_settings, metrics)

        # 9. Store globally for saving
        model_global = model
        vectorizer_global = vectorizer

        # 10. Update context
        context.update({
            'training_session': training_session,
            'classifier_settings': classifier_settings,
            'evaluation_results': metrics,
            'scroll_to': 'step-3'
        })

        messages.success(request, "Model trained successfully!")
        logger.info("Training completed successfully")

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)

        if training_session:
            _handle_training_failure(training_session, error_msg)

        context['error'] = error_msg
        messages.error(request, error_msg)

    return render(request, 'task1.html', context)


def _get_latest_classifier_settings(context):
    """Get the latest classifier settings"""
    classifier_settings = None
    if TextClassifier.objects.exists():
        classifier_settings = TextClassifier.objects.latest('created_at')
        context.update({
            'selected_representation': classifier_settings.representation_type,
            'classifier_settings': classifier_settings
        })

    if not classifier_settings:
        raise TrainingError("No model configured. Please select a model first.")

    return classifier_settings


def _should_use_batch_training(classifier_settings, X_train):
    """Determine if batch training should be used"""
    return (classifier_settings.model_type == 'naive_bayes' and
            classifier_settings.nb_variant == 'gaussian' and
            scipy.sparse.issparse(X_train))


def _update_training_completion(training_session, classifier_settings, metrics):
    """Update training session and classifier settings upon completion"""
    # Update training session
    training_session.status = 'completed'
    training_session.final_accuracy = metrics['accuracy']
    training_session.final_precision = metrics['precision']
    training_session.final_recall = metrics['recall']
    training_session.final_f1 = metrics['f1']
    training_session.end_time = now()
    training_session.duration = training_session.end_time - training_session.start_time
    training_session.save()

    # Update classifier
    classifier_settings.is_trained = True
    classifier_settings.test_accuracy = metrics['accuracy']
    classifier_settings.save()


def _handle_training_failure(training_session, error_msg):
    """Handle training failure by updating session status"""
    training_session.status = 'failed'
    training_session.error_message = error_msg
    training_session.end_time = now()
    training_session.duration = training_session.end_time - training_session.start_time
    training_session.save()


def handle_model_saving(request, context):
    """Handle model saving with improved error handling"""
    global model_global, vectorizer_global

    save_success = False
    save_error = None

    try:
        if DEBUG:
            logger.info("Starting model save process")

        # Validate that we have trained models to save
        if not model_global or not hasattr(model_global, 'is_trained') or not model_global.is_trained:
            raise TrainingError("No trained model available to save")

        if not vectorizer_global:
            raise TrainingError("No vectorizer available to save")

        # Create model directory
        model_dir = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Get representation type and create suffix
        representation_type = context.get('selected_representation', 'tfidf')
        model_suffix = f"{representation_type}"

        # Save model and vectorizer
        model_global.save_classifier(name_suffix=model_suffix)

        vectorizer_filename = os.path.join(model_dir, f"{representation_type}_vectorizer.pkl")
        with open(vectorizer_filename, 'wb') as f:
            pickle.dump(vectorizer_global, f)

        save_success = True
        success_msg = f"Model and {representation_type} vectorizer saved successfully!"
        messages.success(request, success_msg)

        if DEBUG:
            logger.info(f"Model saved with suffix: {model_suffix}")
            logger.info(f"Vectorizer saved to: {vectorizer_filename}")

    except Exception as e:
        save_error = f"Error saving model: {str(e)}"
        messages.error(request, save_error)
        logger.error(save_error)

    # Update context with save status
    context.update({
        'scroll_to': 'step-3',
        'classifier_settings': context.get('classifier_settings'),
        'training_session': context.get('training_session'),
        'evaluation_results': context.get('evaluation_results'),
        'save_success': save_success,
        'save_error': save_error
    })

    return render(request, 'task1.html', context)
