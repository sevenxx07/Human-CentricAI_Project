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
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'load_pre_trained':
            handle_load_pre_trained(request, context)

    return render(request, 'project2_base.html', context)


def handle_load_pre_trained(request, context):
    model_dir = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'models')

    selected_model = request.POST.get('model')
    print('Selected model:', selected_model)

    # Model filename mapping
    model_map = {
        'logistic': {
            'filename': 'LogisticRegressionModel_fullwraper_tfidf.pkl',
            'class': LogisticRegressionModel
        },
        'svm': {
            'filename': 'SVMModel_fullwraper_tfidf.pkl',
            'class': SVMModel
        },
        'naive_bayes': {
            'filename': 'NaiveBayesModel_fullwraper_tfidf.pkl',
            'class': NaiveBayesModel
        }
    }

    if selected_model not in model_map:
        context['error'] = 'Invalid model selected'
        return render(request, 'project2_base.html', context)

    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(model_dir, model_map[selected_model]['filename'])

    print("Looking for files:")
    print(f"  Vectorizer: {vectorizer_path} (exists: {os.path.exists(vectorizer_path)})")
    print(f"  Model: {model_path} (exists: {os.path.exists(model_path)})")

    try:
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Vectorizer loaded successfully")

        # Load the complete wrapper instance (not just sklearn model)
        model_class = model_map[selected_model]['class']
        model_wrapper = model_class.load_model(model_path)
        hyperparameters = model_wrapper.get_hyperparameters()

        text_classifier = TextClassifier.objects.create(
            name=f"{selected_model}_tfidf_classifier",
            model_type=selected_model,
            representation_type='tfidf',
            is_trained=True
        )

        text_classifier.map_hyperparameters(hyperparameters)

        print(f"Found TextClassifier: ID={text_classifier.id}")
        print(f"  Model: {text_classifier.model_type}")
        print(f"  Representation: {text_classifier.representation_type}")
        print(f"  Hyperparameters: C={text_classifier.regularization_c}")


        print(f"✓ Model wrapper loaded: {type(model_wrapper).__name__}")
        print(f"  Is trained: {model_wrapper.is_trained}")
        print(f"  Has sklearn classifier: {model_wrapper.classifier is not None}")

        # For debugging, show the wrapper's attributes
        if hasattr(model_wrapper, 'C'):
            print(f"  Wrapper C parameter: {model_wrapper.C}")
        if hasattr(model_wrapper, 'regularization_c'):
            print(f"  Wrapper regularization_c: {model_wrapper.regularization_c}")

        # Store EVERYTHING in DATA_STORAGE
        DATA_STORAGE.clear()
        DATA_STORAGE.update({
            "vectorizer": vectorizer,
            'model_wrapper': model_wrapper,  # The complete wrapper instance
            'classifier_settings': text_classifier,  # The Django model instance
            'model_loaded_from_file': True,
            'evaluation_results': model_wrapper.metrics
        })

        print(f"✓ DATA_STORAGE updated with keys: {list(DATA_STORAGE.keys())}")

        context.update({
            'message': 'Pre-trained models loaded successfully',
            'model_loaded': True,
            'selected_model': selected_model,
            'text_classifier': text_classifier,
            'model_wrapper': model_wrapper
        })

    except TextClassifier.DoesNotExist:
        error_msg = f'No trained {selected_model} classifier found in database. Please train a model in Task 1 first.'
        print(f"ERROR: {error_msg}")
        context['error'] = error_msg
    except Exception as e:
        error_msg = f'Error loading pre-trained model: {str(e)}'
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        context['error'] = error_msg

    return render(request, 'project2_base.html', context)


def task1_view(request):
    # placeholder — replace with logic later
    return render(request, "task1.html")


def task2_view(request):
    # placeholder — replace with logic later
    return render(request, "task2.html")
