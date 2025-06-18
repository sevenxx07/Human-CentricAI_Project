import pickle
import os

from django.shortcuts import render
from django.template import loader
from django.conf import settings

from project_2.ml_models.LogisticRegressionModel import LogisticRegressionModel
from project_2.ml_models.NaiveBayesModel import NaiveBayesModel
from project_2.ml_models.SVMModel import SVMModel

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

    # TODO only one classifier choicefor now
    model_map = {
        'logistic': {
            'filename': 'LogisticRegressionModel_tfidf.pkl',
            'class': LogisticRegressionModel
        },
        'svm': {
            'filename': 'SVMModel_tfidf.pkl',
            'class': SVMModel
        },
        'naive_bayes': {
            'filename': 'NaiveBayesModel_tfidf.pkl',
            'class': NaiveBayesModel
        }
    }

    if selected_model not in model_map:
        context['error'] = 'Invalid model selected'
        return render(request, 'project2_base.html', context)

    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(model_dir, model_map[selected_model]['filename'])

    print("File exists:", os.path.exists(vectorizer_path))
    print("File size:", os.path.getsize(vectorizer_path) if os.path.exists(vectorizer_path) else "N/A")
    print("Size (bytes):",
          os.path.getsize(vectorizer_path) if os.path.exists(vectorizer_path) else "File does not exist")

    print("Looking for model in:", model_dir)
    print("Vectorizer path:", vectorizer_path)
    print("Model path:", model_path)

    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        model_class = model_map[selected_model]['class']
        print("Model class:", model_class)
        model_instance = model_class()
        clf = model_instance.load_model(model_path)
        print("Model loaded successfully:", clf)

        # TODO how to transfer it to task two?
        DATA_STORAGE.update({
            "vectorizer": vectorizer,
            'clf': clf
        })

        context.update({
            'message': 'Pre-trained models loaded successfully',
            'model_loaded': True,
            'selected_model': selected_model
        })

    except Exception as e:
        context['error'] = f'Error loading pre-trained model: {str(e)}'
    return render(request, 'task1.html', context)


def task1_view(request):
    # placeholder — replace with logic later
    return render(request, "task1.html")


def task2_view(request):
    # placeholder — replace with logic later
    return render(request, "task2.html")
