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
    context = {'model_trained': False, 'error': None, 'scroll_to': request.POST.get('scroll_to')}

    classifier = TextClassifier.objects.filter(is_trained=True).first()
    if classifier:
        context['model_trained'] = True
        context['classifier'] = classifier

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'select_model':
            return handle_model_selection(request, context)
        elif action == 'train_model':
            return handle_model_training(request, context)
        elif action == 'test_model':
            pass
            # return handle_model_testing(request, context)

    return render(request, "task1.html", context)


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


def handle_model_training(request, context, upload_dir="./data/cleaned_imdb_reviews.csv"):
    """Handle model training process"""
    context['scroll_to'] = request.POST.get('scroll_to')

    try:
        # Get the dataset
        df = pd.read_csv(upload_dir)

        # Get classifier
        classifier = TextClassifier.objects.filter(model_type=request.POST.get('model')).first()
        if not classifier:
            context['error'] = "No model configuration found. Please select a model first."
            return render(request, 'task1.html', context)

        # Create training session
        training_session = TrainingSession.objects.create(
            classifier=classifier,
            status='running'
        )

        # Prepare data
        text_column = 'review' if 'review' in df.columns else df.columns[0]
        label_column = 'sentiment' if 'sentiment' in df.columns else 'label'

        X = df[text_column].astype(str)
        y = df[label_column]

        # Convert string labels to binary if needed
        if y.dtype == 'object':
            y = (y == 'positive').astype(int)

        # Split data
        test_size = float(request.POST.get('test_size', 20)) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Text representation - now properly using the classifier's representation_type
        if classifier.representation_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        elif classifier.representation_type == 'glove':
            # For GloVe, you would typically use pre-trained embeddings
            # This is a placeholder - you'll need to implement GloVe embedding
            raise NotImplementedError("GloVe representation not implemented yet")
        elif classifier.representation_type == 'sbert':
            # For SBERT, you would use sentence-transformers
            # This is a placeholder - you'll need to implement SBERT
            raise NotImplementedError("SBERT representation not implemented yet")
        else:  # default to bow if representation_type is not recognized
            vectorizer = CountVectorizer(max_features=10000, stop_words='english')

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Initialize model
        if classifier.model_type == 'logistic':
            pass
            # TODO Logistic_regression module is full of error
            # model = LogisticRegression(C=classifier.regularization_c, random_state=42)
        elif classifier.model_type == 'svm':
            model = SVMClassifier(kernel=classifier.kernel, random_state=42)
        elif classifier.model_type == 'naive_bayes':
            pass

        model.train(X_train_vec, y_train)

        # Predictions
        y_pred = model.predict(X_test_vec)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Update classifier
        classifier.is_trained = True
        classifier.test_accuracy = accuracy * 100
        classifier.save()
        classifier.save_model(model, vectorizer)

        # Update training session
        training_session.status = 'completed'
        training_session.end_time = datetime.now()
        training_session.training_samples = len(X_train)
        training_session.validation_samples = len(X_test)
        training_session.final_accuracy = accuracy * 100
        training_session.final_precision = precision * 100
        training_session.final_recall = recall * 100
        training_session.final_f1 = f1 * 100
        training_session.save()

        context['model_trained'] = True
        context['train_success'] = True
        context['accuracy'] = round(accuracy * 100, 2)
        context['precision'] = round(precision * 100, 2)
        context['recall'] = round(recall * 100, 2)
        context['f1'] = round(f1 * 100, 2)
        context['classifier'] = classifier
        context['training_session'] = training_session
        context['scroll_to'] = 'step-3'

        messages.success(request, f"Model trained successfully! Test accuracy: {accuracy * 100:.2f}%")

    except Exception as e:
        if 'training_session' in locals():
            training_session.status = 'failed'
            training_session.error_message = str(e)
            training_session.save()
        context['error'] = f"Error training model: {str(e)}"

    return render(request, 'task1.html', context)