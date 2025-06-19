import os
import pickle
from pathlib import Path
from abc import ABC, abstractmethod

from django.conf import settings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ClassifierWrapper(ABC):
    """
    Abstract base class for all classifier models with common functionality.
    Provides a unified interface for training, prediction, evaluation, and model persistence.
    """

    def __init__(self, random_state=42, verbose=True):
        """
        Initialize common classifier parameters.

        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility
        verbose : bool, default=True
            Enable verbose output
        """
        self.random_state = random_state
        self.verbose = verbose
        self.is_trained = False
        self.classifier = None

    def train(self, X_train, y_train):
        """
        Train the classifier on the provided data.

        Parameters:
        -----------
        X_train : array-like
            Training feature vectors
        y_train : array-like
            Training target values

        Returns:
        --------
        self : ClassifierWrapper
            Returns self for method chaining
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not initialized. Check child class implementation.")

        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        if self.verbose:
            print(f"Model {type(self).__name__} trained successfully")

        return self

    def partial_fit(self, X, y, classes=None):
        """
        Incremental fit on a batch of samples (for online learning).
        Only available for classifiers that support incremental learning.

        Parameters:
        -----------
        X : array-like
            Feature vectors
        y : array-like
            Target values
        classes : array-like, optional
            List of all possible classes (required for first call)

        Returns:
        --------
        self : ClassifierWrapper
            Returns self for method chaining
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not initialized. Check child class implementation.")

        if not hasattr(self.classifier, 'partial_fit'):
            raise NotImplementedError(
                f"Partial fit not available for {type(self.classifier).__name__}")

        self.classifier.partial_fit(X, y, classes=classes)
        self.is_trained = True

        if self.verbose:
            print(f"Partial fit completed for {type(self).__name__}")

        return self

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        -----------
        X : array-like
            Feature vectors to predict

        Returns:
        --------
        predictions : array
            Predicted class labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        if self.classifier is None:
            raise RuntimeError("Classifier not initialized")

        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Return probability estimates for the test data.

        Parameters:
        -----------
        X : array-like
            Feature vectors to predict

        Returns:
        --------
        probabilities : array
            Probability estimates for each class
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        if self.classifier is None:
            raise RuntimeError("Classifier not initialized")

        if not hasattr(self.classifier, 'predict_proba'):
            raise NotImplementedError(
                f"Probability prediction not available for {type(self.classifier).__name__}")

        return self.classifier.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data and return multiple metrics.

        Parameters:
        -----------
        X_test : array-like
            Test feature vectors
        y_test : array-like
            True labels for test data

        Returns:
        --------
        metrics : dict
            Dictionary containing accuracy, precision, recall, and F1 score
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        y_pred = self.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
        }

        if self.verbose:
            print(f"Evaluation metrics for {type(self).__name__}:")
            for metric, value in metrics.items():
                print(f"  {metric.capitalize()}: {value:.4f}")

        return metrics

    def save_classifier(self, file_path=None, name_suffix=None):
        """
        Save the trained model to a file.

        Parameters:
        -----------
        file_path : str or Path, optional
            Custom path to save the model. If None, uses default location.
        name_suffix : str, optional
            Suffix to add to the default filename
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")

        if self.classifier is None:
            raise RuntimeError("No classifier to save")

        # Compute the default file path if none is provided
        if file_path is None:
            model_dir = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'models')
            suffix = f"_{name_suffix}" if name_suffix else ""
            filename = f"{type(self).__name__}{suffix}.pkl"
            file_path = os.path.join(model_dir, filename)

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.classifier, f)

            if self.verbose:
                print(f"Model saved to: {file_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to save model: {str(e)}")

    @classmethod
    def load_model(cls, file_path=None, name_suffix=None):
        """
        Load a saved model from file and return a wrapped instance.

        Parameters:
        -----------
        file_path : str or Path, optional
            Path to the saved model. If None, uses default location.
        name_suffix : str, optional
            Suffix of the model file to load

        Returns:
        --------
        instance : ClassifierWrapper
            Loaded classifier instance with the model restored
        """
        # Compute the default file path if none is provided
        if file_path is None:
            model_dir = os.path.join(settings.BASE_DIR, 'project_2', 'data', 'models')
            suffix = f"_{name_suffix}" if name_suffix else ""
            filename = f"{cls.__name__}{suffix}.pkl"
            file_path = os.path.join(model_dir, filename)

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                loaded_classifier = pickle.load(f)

            # Create a new instance of the wrapper class
            # Note: This assumes the child class can be initialized with default parameters
            instance = cls()
            instance.classifier = loaded_classifier
            instance.is_trained = True

            if instance.verbose:
                print(f"Model loaded from: {file_path}")

            return instance

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")