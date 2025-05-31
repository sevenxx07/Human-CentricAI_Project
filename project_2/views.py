import pickle
import numpy as np
import pandas as pd
import os
import ast

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import JsonResponse 
from django.conf import settings 

#from .ML_models.Pre_Processing import clean_text
from .ML_models.representation import load_dataset, tfidf_representation
from .ML_models.Logistic_regression import train_classifier, evaluate_classifier, save_model, load_model 

from sklearn.model_selection import train_test_split

DATA_STORAGE = {}

def index(request):
    template = loader.get_template("project2_base.html")
    context = {}

    global DATA_STORAGE

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'launch_training':
            handle_launch_training(request, context)
        elif action == 'load_pre_trained': 
            handle_load_pre_trained(request, context)
        elif action == 'classify text': # If we want to implement it later
            pass

    return render(request, 'project2_base.html', context)


# Launch the training of the classifier 
def handle_launch_training(request, context):
    context['training_started'] = True
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'project_2', 'ML_models', 'cleaned_imdb_reviews.csv')
    
    df = pd.read_csv(data_path)
    df = df[df['review'].notna()]

    df['review']= df['review'].apply(ast.literal_eval)
    print(f"HEAD:",df.head())

    df['review'] = df['review'].apply(lambda tokens: " ".join(tokens))
    
    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()
    vectors, encoder = tfidf_representation(texts)

    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size= 0.2, random_state=42)

    # # Inside your Django view or a helper function where you load vectors and labels

    clf = train_classifier(X_train, y_train)
    accuracy = evaluate_classifier(clf, X_test,y_test)
    context['accuracy'] = accuracy 

    DATA_STORAGE.update({
        'df': df,
        'X': vectors,
        'y': labels,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'clf': clf
    })


def handle_load_pre_trained(request, context):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(BASE_DIR, 'project_2', 'ML_models')

    vectorizer_path = os.path.join(model_dir, 'tfidf_encoder.pkl')
    model_path = os.path.join(model_dir, 'tfidf_classifier.pkl')


    print("Looking for model in:", model_dir)
    print("Vectorizer path:", vectorizer_path)
    print("Model path:", model_path)

    try: 
        with open(vectorizer_path, 'rb') as f: 
            vectorizer = pickle.load(f)
        clf = load_model(model_path)
    except FileNotFoundError: 
        context['error'] = 'Pre-trained models not found'
        return 
    
    DATA_STORAGE.update({
        "vectorizer": vectorizer, 
        'clf': clf
    })
    context['message'] = 'Pre-trained models loaded successfully'

    
def handle_classify_text(request, context):
    pass


