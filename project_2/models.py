from django.db import models

from django.db import models
import pickle
import os
from django.conf import settings


class TextClassifier(models.Model):
    """Model to store text classifier information"""

    REPRESENTATION_CHOICES = [
        ('tfidf', 'TF-IDF'),
        ('glove', 'GloVe'),
        ('sbert', 'SBERT'),
    ]

    name = models.CharField(max_length=100, default="IMDB Sentiment Classifier")
    model_type = models.CharField(max_length=50, default="logistic")
    representation_type = models.CharField(
        max_length=50,
        choices=REPRESENTATION_CHOICES,
        default="tfidf"
    )
    is_trained = models.BooleanField(default=False)
    train_accuracy = models.FloatField(null=True, blank=True)
    test_accuracy = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Hyperparameters
    regularization_c = models.FloatField(default=1.0)  # For Logistic Regression and SVM
    kernel = models.CharField(max_length=50, default="linear", null=True, blank=True)  # For SVM
    alpha = models.FloatField(default=1.0, null=True, blank=True)  # For Naive Bayes
    def __str__(self):
        return f"{self.name} - {self.model_type}"

    def save_model(self, model, vectorizer=None):
        """Save trained model and vectorizer to files"""
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Save the classifier model
        model_path = os.path.join(model_dir, f'classifier_{self.id}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save the vectorizer if provided
        if vectorizer:
            vectorizer_path = os.path.join(model_dir, f'vectorizer_{self.id}.pkl')
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)

    def load_model(self):
        """
        Load trained model and vectorizer from files

        Returns: sklearn model, vectorization

        """
        model_dir = os.path.join(settings.DATA_ROOT, 'models')

        try:
            # Load classifier
            model_path = os.path.join(model_dir, f'classifier_{self.id}.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Load vectorizer
            vectorizer_path = os.path.join(model_dir, f'vectorizer_{self.id}.pkl')
            vectorizer = None
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)

            return model, vectorizer
        except FileNotFoundError:
            return None, None


class TrainingSession(models.Model):
    """Model to track training sessions"""
    classifier = models.ForeignKey(TextClassifier, on_delete=models.CASCADE)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], default='running')

    # Training metrics
    training_samples = models.IntegerField(null=True, blank=True)
    validation_samples = models.IntegerField(null=True, blank=True)
    final_accuracy = models.FloatField(null=True, blank=True)
    final_precision = models.FloatField(null=True, blank=True)
    final_recall = models.FloatField(null=True, blank=True)
    final_f1 = models.FloatField(null=True, blank=True)

    error_message = models.TextField(blank=True)

    def __str__(self):
        return f"Training Session {self.id} - {self.status}"


class TrainedModelData(models.Model):
    name = models.CharField(max_length=100)
    vectorizer = models.BinaryField()
    classifier = models.BinaryField()
    accuracy = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name