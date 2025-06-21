import pickle
import os

from django.shortcuts import render
from django.template import loader
from django.conf import settings

from project_2.ml_models.LogisticRegressionModel import LogisticRegressionModel
from project_2.ml_models.NaiveBayesModel import NaiveBayesModel
from project_2.ml_models.SVMModel import SVMModel

from project_2.models import TextClassifier

DEBUG = True  # Set to False in production
model_global = None  # Global variable to hold the model instance
vectorizer_global = None  # Global variable to hold the vectorizer instance
DATA_STORAGE = {}


def index(request):
    template = loader.get_template("project2_base.html")
    context = {
        'selected_model': 'logistic',  # Default selection
        'model_loaded': bool(DATA_STORAGE.get('model_wrapper')),
        'loaded_model_info': get_loaded_model_info(),
        'evaluation_results': DATA_STORAGE.get('evaluation_results')
    }

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'load_pre_trained':
            context['selected_model'] = request.POST.get('model', 'logistic')
            handle_load_pre_trained(request, context)

    return render(request, 'project2_base.html', context)


def get_loaded_model_info():
    """Get information about currently loaded model"""
    if not DATA_STORAGE.get('model_wrapper'):
        return None

    classifier_settings = DATA_STORAGE.get('classifier_settings')
    model_wrapper = DATA_STORAGE.get('model_wrapper')

    if not classifier_settings:
        return None

    return {
        'model_type': classifier_settings.model_type,
        'representation_type': classifier_settings.representation_type,
        'is_trained': model_wrapper.is_trained if model_wrapper else False,
        'hyperparameters': classifier_settings.get_hyperparameters()
    }


def handle_load_pre_trained(request, context):
    model_dir = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'models')

    selected_model = request.POST.get('model')
    if DEBUG:
        print(f'Selected model: {selected_model}')

    # Model filename mapping
    model_map = {
        'logistic': {
            'filename': 'LogisticRegressionModel_fullwraper_tfidf.pkl',
            'class': LogisticRegressionModel,
            'display_name': 'Logistic Regression'
        },
        'svm': {
            'filename': 'SVMModel_fullwraper_tfidf.pkl',
            'class': SVMModel,
            'display_name': 'Support Vector Machine'
        },
        'naive_bayes': {
            'filename': 'NaiveBayesModel_fullwraper_tfidf.pkl',
            'class': NaiveBayesModel,
            'display_name': 'Naive Bayes'
        }
    }

    if selected_model not in model_map:
        context['error'] = 'Invalid model selected'
        return

    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(model_dir, model_map[selected_model]['filename'])

    if DEBUG:
        print("Looking for files:")
        print(f"  Vectorizer: {vectorizer_path} (exists: {os.path.exists(vectorizer_path)})")
        print(f"  Model: {model_path} (exists: {os.path.exists(model_path)})")

    # Check if files exist
    if not os.path.exists(vectorizer_path):
        context['error'] = f'Vectorizer file not found: {vectorizer_path}. Please train a model in Task 1 first.'
        return

    if not os.path.exists(model_path):
        context['error'] = f'Model file not found: {model_path}. Please train a model in Task 1 first.'
        return

    try:
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        if DEBUG:
            print("✓ Vectorizer loaded successfully")

        # Load the complete wrapper instance (not just sklearn model)
        model_class = model_map[selected_model]['class']
        model_wrapper = model_class.load_model(model_path)
        hyperparameters = model_wrapper.get_hyperparameters()

        # Create TextClassifier instance for consistency
        text_classifier = TextClassifier.objects.create(
            name=f"{selected_model}_tfidf_classifier",
            model_type=selected_model,
            representation_type='tfidf',
            is_trained=True
        )

        text_classifier.map_hyperparameters(hyperparameters)

        if DEBUG:
            print(f"✓ Created TextClassifier: ID={text_classifier.id}")
            print(f"  Model: {text_classifier.model_type}")
            print(f"  Representation: {text_classifier.representation_type}")
            print(f"✓ Model wrapper loaded: {type(model_wrapper).__name__}")
            print(f"  Is trained: {model_wrapper.is_trained}")
            print(f"  Has sklearn classifier: {model_wrapper.classifier is not None}")

        # Convert metrics to percentage format if they exist
        evaluation_results = None
        if hasattr(model_wrapper, 'metrics') and model_wrapper.metrics:
            evaluation_results = {}
            for key, value in model_wrapper.metrics.items():
                if value is not None:
                    # Convert to percentage
                    evaluation_results[key] = value * 100
            if DEBUG:
                print(f"✓ Evaluation results: {evaluation_results}")

        # Store EVERYTHING in DATA_STORAGE
        DATA_STORAGE.clear()
        DATA_STORAGE.update({
            "vectorizer": vectorizer,
            'model_wrapper': model_wrapper,  # The complete wrapper instance
            'classifier_settings': text_classifier,  # The Django model instance
            'model_loaded_from_file': True,
            'evaluation_results': evaluation_results  # Converted to percentage
        })

        if DEBUG:
            print(f"✓ DATA_STORAGE updated with keys: {list(DATA_STORAGE.keys())}")
            print(
                f"  Baseline accuracy available: {evaluation_results.get('accuracy') if evaluation_results else 'None'}")

        # Update context with success info
        context.update({
            'message': f'{model_map[selected_model]["display_name"]} model loaded successfully!',
            'model_loaded': True,
            'loaded_model_info': get_loaded_model_info(),
            'evaluation_results': evaluation_results
        })

    except Exception as e:
        error_msg = f'Error loading pre-trained model: {str(e)}'
        if DEBUG:
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
        context['error'] = error_msg


def task1_view(request):
    # placeholder — replace with logic later
    return render(request, "task1.html")


def task2_view(request):
    # placeholder — replace with logic later
    return render(request, "task2.html")