from django.db import models
from pathlib import Path


class TextClassifier(models.Model):
    """Model to store text classifier information"""
    MODEL_TYPES = {
        'svm': 'Support Vector Machine',
        'logistic': 'Logistic Regression',
        'naive_bayes': 'Naive Bayes'
    }

    REPRESENTATIONS = {
        'tfidf': 'TF-IDF',
        'glove': 'GloVe',
        'sbert': 'SBERT'
    }

    name = models.CharField(max_length=100, default="classifier")
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES.items(), default="logistic")
    representation_type = models.CharField(max_length=50, choices=REPRESENTATIONS.items(), default="tfidf")

    is_trained = models.BooleanField(default=False)
    test_accuracy = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Hyperparameters (simplified null/blank handling)
    regularization_c = models.FloatField(default=1.0, blank=True)
    max_iter = models.IntegerField(default=1000, blank=True)  # Used by both Logistic and SVM
    solver = models.CharField(max_length=20, default="lbfgs", blank=True)
    penalty = models.CharField(max_length=20, default="l2", blank=True)
    nb_variant = models.CharField(max_length=20, default="gaussian", blank=True)
    fit_prior = models.BooleanField(default=True, blank=True)
    alpha = models.FloatField(default=1.0, blank=True)
    kernel = models.CharField(max_length=50, default="linear", blank=True)
    gamma = models.CharField(max_length=20, default="scale", blank=True)

    def __str__(self):
        return f"{self.model_type}_{self.representation_type}_classifier"

    def map_hyperparameters(self, hyperparams):
        """Map hyperparameters from dict to model fields"""
        for field in self._meta.get_fields():
            if not field.is_relation and field.name in hyperparams:
                setattr(self, field.name, hyperparams[field.name])
        self.save()

    def create_model_instance(self):
        """Create appropriate model wrapper instance"""
        model_params = {
            'random_state': 42,
            'verbose': False
        }

        if self.model_type == 'svm':
            from project_2.ml_models.SVMModel import SVMModel
            model_params.update({
                'kernel': self.kernel,
                'C': self.regularization_c,
                'gamma': self.gamma,
                'max_iter': self.max_iter
            })
            return SVMModel(**model_params)

        elif self.model_type == 'logistic':
            from project_2.ml_models.LogisticRegressionModel import LogisticRegressionModel
            model_params.update({
                'C': self.regularization_c,
                'max_iter': self.max_iter,  # This was missing the proper parameter passing
                'solver': self.solver,
                'penalty': self.penalty  # Also add penalty parameter
            })
            return LogisticRegressionModel(**model_params)

        elif self.model_type == 'naive_bayes':
            from project_2.ml_models.NaiveBayesModel import NaiveBayesModel
            model_params.update({
                'variant': self.nb_variant,
                'alpha': self.alpha,
                'fit_prior': self.fit_prior  # Also add fit_prior parameter
            })
            return NaiveBayesModel(**model_params)

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def get_hyperparameters(self):
        """Return hyperparameters as dict"""
        relevant_fields = {
        'logistic': ['regularization_c', 'max_iter', 'solver', 'penalty'],
        'svm': ['regularization_c', 'max_iter', 'kernel', 'gamma'],
        'naive_bayes': ['nb_variant', 'alpha', 'fit_prior']
        }

        # Get the relevant fields for this model type
        model_fields = relevant_fields.get(self.model_type, [])

        # Return only the hyperparameters relevant to this model type
        params = {}
        for field_name in model_fields:
            if hasattr(self, field_name):
                params[field_name] = getattr(self, field_name)

        return params

class TrainingSession(models.Model):
    """Model to track training sessions"""
    STATUSES = {
        'pending': 'Pending',
        'running': 'Running',
        'completed': 'Completed',
        'failed': 'Failed'
    }
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUSES.items(), default='pending')

    # Metrics
    final_accuracy = models.FloatField(null=True, blank=True)
    final_precision = models.FloatField(null=True, blank=True)
    final_recall = models.FloatField(null=True, blank=True)
    final_f1 = models.FloatField(null=True, blank=True)

    error_message = models.TextField(blank=True)
    notes = models.TextField(blank=True)
    duration = models.DurationField(null=True, blank=True)

    class Meta:
        ordering = ['-start_time']