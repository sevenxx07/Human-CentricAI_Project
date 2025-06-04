import ast
import pandas as pd
import pickle

from django.shortcuts import render
from django.contrib import messages
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from django.utils.timezone import now
from pbl.settings import DATA_ROOT
from project_2.ML_models.Representation import tfidf_representation, sbert_representation, glove_representation
from project_2.models import TextClassifier, TrainingSession

DEBUG = True  # Set to False in production
model_global = None  # Global variable to hold the model instance
vectorizer_global = None  # Global variable to hold the vectorizer instance


def index(request):
    """Main view for the text classification interface"""
    context = {
        'error': None,
        'scroll_to': request.POST.get('scroll_to', 'step-1'),
        'selected_representation': 'tfidf',  # Default value
        'classifier_settings': None,
        'training_session': None,
        'MODEL_TYPES': TextClassifier.MODEL_TYPES,  # Add these
        'REPRESENTATIONS': TextClassifier.REPRESENTATIONS  # Add these
    }

    if DEBUG:
        print("\n=== DEBUG: index() called ===")
        print(f"Initial context: {context}")

    # Get the most recent classifier if exists
    if TextClassifier.objects.exists():
        latest_classifier = TextClassifier.objects.latest('created_at')
        context.update({
            'selected_representation': latest_classifier.representation_type,
            'classifier_settings': latest_classifier
        })

    if request.method == 'POST':
        action = request.POST.get('action')
        if DEBUG:
            print(f"POST action received: {action}")

        if action == 'select_model':
            return handle_model_selection(request, context)
        elif action == 'train_model':
            return handle_model_training(request, context)
        elif action == 'save_model':
            return handle_model_saving(request, context)

    return render(request, "task1.html", context)


def update_classifier_data(model_type, request, classifier_data):
    """Update classifier data based on the model type and request parameters."""

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


def handle_model_selection(request, context):
    """Handle model type and hyperparameter selection"""

    try:
        if DEBUG:
            print("\n=== DEBUG: handle_model_selection() ===")
            print(f"Incoming POST data: {request.POST}")

        model_type = request.POST.get('model')
        representation_type = request.POST.get('representation', 'tfidf')

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
            print(f"Model configured successfully. Updated context: {context}")

    except Exception as e:
        context['error'] = f"Error configuring model: {str(e)}"
        messages.error(request, context['error'])
        if DEBUG:
            print(f"Error in model selection: {context['error']}")

    return render(request, 'task1.html', context)


def load_data(data_path):
    """Load and preprocess the dataset"""

    df = pd.read_csv(data_path)
    df = df[df['review'].notna()]
    df['review'] = df['review'].apply(ast.literal_eval)
    df['review'] = df['review'].apply(lambda tokens: " ".join(tokens))
    return df['review'].tolist(), df['sentiment'].tolist()


def handle_model_training(request, context, data_path=None):
    """Handle model training process following Jupyter notebook steps"""

    global model_global
    global vectorizer_global

    if DEBUG:
        print("\n=== DEBUG: handle_model_training() ===")
        print(f"Initial context: {context}")

    training_session = None
    context.update({
        'scroll_to': 'step-3'
    })

    if data_path is None:  # NOTE maybe configure better
        data_path = f"{DATA_ROOT}/project2_data/cleaned_imdb_reviews.csv"

    try:
        classifier_settings = None
        if TextClassifier.objects.exists():
            classifier_settings = TextClassifier.objects.latest('created_at')
            context.update({
                'selected_representation': classifier_settings.representation_type,
                'classifier_settings': classifier_settings
            })

        if not classifier_settings:
            raise ValueError("No model configured. Please select a model first.")
        if DEBUG:
            print(f"Starting training for model: {classifier_settings}")

        training_session = TrainingSession.objects.create(status='running')
        context.update({
            'training_session': training_session,
        })

        if DEBUG:
            print(f"Training session created: {training_session}")

        # 1. Load and prepare data
        texts, labels = load_data(data_path)
        training_session.status = 'data loaded'
        training_session.save()
        if DEBUG:
            print(f"Loaded {len(texts)} texts and {len(labels)} labels.")

        # 2. Create text representation
        representation_type = classifier_settings.representation_type
        if representation_type == 'tfidf':
            X, vectorizer = tfidf_representation(texts)
        elif representation_type == 'sbert':
            X, vectorizer = sbert_representation(texts)
        elif representation_type == 'glove':
            X, vectorizer = glove_representation(texts)
        else:
            raise ValueError(f"Unknown representation type: {representation_type}")
        y = labels

        training_session.status = 'data vectorized'
        training_session.save()
        if DEBUG:
            print(f"Text representation created. Sample vector shape: {X.shape}")

        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        training_session.status = 'data split'
        training_session.save()

        # 4. Create and train model
        training_session.status = 'training model'
        model = classifier_settings.create_model_instance()
        model.train(X_train, y_train)
        training_session.status = 'model trained'
        training_session.save()
        if DEBUG:
            print(f"Model trained successfully: {model}")

        # 5. Evaluation
        y_pred = model.predict(X_test)

        # Calculate and store metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred) * 100,
            'precision': precision_score(y_test, y_pred, average='weighted') * 100,
            'recall': recall_score(y_test, y_pred, average='weighted') * 100,
            'f1': f1_score(y_test, y_pred, average='weighted') * 100
        }
        if DEBUG:
            print(f"Evaluation metrics: {metrics}")

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

        model_global = model  # Store the trained model globally
        vectorizer_global = vectorizer  # Store the vectorizer globally

        # Prepare context for GUI
        context.update({
            'training_session': training_session,
            'classifier_settings': classifier_settings,
            'evaluation_results': metrics,
            'scroll_to': 'step-3'
        })

        messages.success(request, "Model trained successfully!")

    except Exception as e:
        error_msg = f"Error training model after {training_session.status}: {str(e)}"
        if training_session:
            training_session.status = 'failed'
            training_session.error_message = error_msg
            training_session.end_time = now()
            training_session.duration = training_session.start_time - training_session.end_time
            training_session.save()
            if DEBUG:
                print(f"Training failed: {error_msg}")
                print(f"Failed session state: {training_session.__dict__}")

        context['error'] = error_msg
        messages.error(request, error_msg)

    return render(request, 'task1.html', context)


def handle_model_saving(request, context):
    global model_global
    global vectorizer_global

    # Add these new variables to track save status
    save_success = False
    save_error = None

    try:
        if DEBUG:
            print("\nDEBUG: Starting model save process")

        if not model_global or not hasattr(model_global, 'is_trained') or not model_global.is_trained:
            raise ValueError("No trained model available to save")
        if not vectorizer_global:
            raise ValueError("No vectorizer available to save")

        representation_type = context.get('selected_representation', 'tfidf')
        model_suffix = f"{representation_type}"

        model_global.save_classifier(name_suffix=model_suffix)
        vectorizer_filename = f"{DATA_ROOT}/project2_data/{representation_type}_vectorizer.pkl"
        with open(vectorizer_filename, 'wb') as f:
            pickle.dump(vectorizer_global, f)

        messages.success(request, f"Model and {representation_type} vectorizer saved successfully!")
        save_success = True

        if DEBUG:
            print(f"Model saved with suffix: {model_suffix}")
            print(f"Vectorizer saved to: {vectorizer_filename}")

    except Exception as e:
        error_msg = f"Error saving model: {str(e)}"
        messages.error(request, error_msg)
        save_error = error_msg
        if DEBUG:
            print(f"Error saving model: {error_msg}")

    # Update context with save status
    context.update({
        'scroll_to': 'step-3',
        'classifier_settings': context.get('classifier_settings'),
        'training_session': context.get('training_session'),
        'evaluation_results': context.get('evaluation_results'),
        'save_success': save_success,  # New context variable
        'save_error': save_error  # New context variable
    })

    return render(request, 'task1.html', context)
