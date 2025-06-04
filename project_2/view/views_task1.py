import os
from datetime import datetime

import pandas as pd
from django.shortcuts import render
from django.template import loader
from django.contrib import messages
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from project_2.ML_models.SVMClassifier import SVMClassifier
from project_2.models import TextClassifier, TrainingSession


def index(request):
    # Initialize context with default values
    context = {
        'model_trained': False,
        'error': None,
        'scroll_to': request.POST.get('scroll_to', 'step-1'),
        'selected_model': TextClassifier.objects.all().last().model_type if TextClassifier.objects.exists() else None,
        'selected_representation': TextClassifier.objects.all().last().representation if TextClassifier.objects.exists() else None,
        'classifier': TextClassifier.objects.all().last(),
        'training_session': None
    }

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'select_model':
            return handle_model_selection(request, context)
        elif action == 'train_model':
            return handle_model_training(request, context)

    return render(request, "task1.html", context)


def evaluate_model(y_test, y_pred):
    """Calculate and return evaluation metrics"""
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }


def handle_model_selection(request, context):
    """Handle model type and hyperparameter selection"""
    context['scroll_to'] = request.POST.get('scroll_to')

    try:
        model_type = request.POST.get('model')
        representation_type = request.POST.get('representation', 'tfidf')

        # Get or create classifier
        classifier, created = TextClassifier.objects.get_or_create(
            model_type=model_type,
            defaults={
                'name': f"{model_type.title()} Classifier",
                'representation_type': representation_type
            }
        )

        # Update all fields
        classifier.representation_type = representation_type

        # Update hyperparameters based on model type
        if model_type == 'logistic':
            classifier.regularization_c = float(request.POST.get('C', 1.0))
            classifier.kernel = None
            classifier.alpha = None
        elif model_type == 'svm':
            classifier.regularization_c = float(request.POST.get('C', 1.0))
            classifier.kernel = request.POST.get('kernel', 'linear')
            classifier.alpha = None
        elif model_type == 'naive_bayes':
            classifier.alpha = float(request.POST.get('alpha', 1.0))
            classifier.regularization_c = None
            classifier.kernel = None

        classifier.save()

        context['model_selected'] = True
        context['selected_model'] = model_type
        context['selected_representation'] = representation_type
        context['classifier'] = classifier
        context['scroll_to'] = 'step-1'  # Changed to step-1 to match your HTML

        messages.success(request, f"{model_type.title()} model configured successfully!")

    except Exception as e:
        context['error'] = f"Error configuring model: {str(e)}"

    return render(request, 'task1.html', context)


def handle_model_training(request, context, data_path="./data/cleaned_imdb_reviews.csv"):
    """Handle model training process"""
    try:
        # Get classifier from request or context
        classifier_id = request.POST.get('classifier_id')
        if classifier_id:
            classifier = TextClassifier.objects.get(id=classifier_id)
        else:
            classifier = context.get('classifier')

        if not classifier:
            raise ValueError("No model configured. Please select a model first.")

        # Update context with correct classifier info
        context.update({
            'classifier': classifier,
            'selected_model': classifier.model_type,
            'selected_representation': classifier.representation_type
        })

        # Load dataset
        df = pd.read_csv(data_path)
        text_column = 'review' if 'review' in df.columns else df.columns[0]
        label_column = 'sentiment' if 'sentiment' in df.columns else 'label'

        X = df[text_column].astype(str)
        y = df[label_column]

        # Convert string labels to binary if needed
        if y.dtype == 'object':
            y = (y == 'positive').astype(int)

        # Split data
        train_size = int(request.POST.get('train_size', 10000))
        if train_size > len(X):
            train_size = len(X)

        X = X[:train_size]
        y = y[:train_size]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create training session
        training_session = TrainingSession.objects.create(
            classifier=classifier,
            status='running',
            training_samples=len(X_train),
            validation_samples=len(X_test)
        )

        # Text representation
        if classifier.representation_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        elif classifier.representation_type == 'glove':
            raise NotImplementedError("GloVe representation not implemented yet")
        elif classifier.representation_type == 'sbert':
            raise NotImplementedError("SBERT representation not implemented yet")
        else:
            vectorizer = CountVectorizer(max_features=10000, stop_words='english')

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Initialize and train model based on classifier type
        if classifier.model_type == 'svm':
            model = SVMClassifier(kernel=classifier.kernel, random_state=42)
            model.train(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
        elif classifier.model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=classifier.regularization_c, random_state=42, max_iter=1000)
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
        elif classifier.model_type == 'naive_bayes':
            from sklearn.naive_bayes import MultinomialNB
            model = MultinomialNB(alpha=classifier.alpha)
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
        else:
            raise NotImplementedError(f"{classifier.model_type} not implemented yet")

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        # Update classifier and training session
        classifier.is_trained = True
        classifier.test_accuracy = metrics['accuracy'] * 100
        classifier.save()
        classifier.save_model(model, vectorizer)

        training_session.status = 'completed'
        training_session.final_accuracy = metrics['accuracy'] * 100
        training_session.final_precision = metrics['precision'] * 100
        training_session.final_recall = metrics['recall'] * 100
        training_session.final_f1 = metrics['f1'] * 100
        training_session.end_time = datetime.now()
        training_session.save()

        # Update context
        context.update({
            'model_trained': True,
            'training_session': training_session,
            'classifier': classifier,
            'selected_model': classifier.model_type,
            'selected_representation': classifier.representation_type,
            'scroll_to': request.POST.get('scroll_to', 'step-2')
        })

        messages.success(request, f"Model trained successfully! Accuracy: {metrics['accuracy'] * 100:.2f}%")

    except Exception as e:
        if 'training_session' in locals():
            training_session.status = 'failed'
            training_session.error_message = str(e)
            training_session.end_time = datetime.now()
            training_session.save()
            context['training_session'] = training_session

        context['error'] = f"Error training model: {str(e)}"
        messages.error(request, context['error'])

    return render(request, 'task1.html', context)
