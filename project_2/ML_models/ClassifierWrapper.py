import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import cross_val_score
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

from pbl.settings import DATA_ROOT


class ClassifierWrapper(ABC):
    """
    Abstract base class for all classifier models with common functionality.
    """

    def __init__(self, random_state=42, verbose=True):
        """
        Initialize common classifier parameters.
        """
        self.random_state = random_state
        self.verbose = verbose
        self.is_trained = False
        self.classifier = None

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the classifier on the provided data.
        Must be implemented by child classes.
        """
        pass

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
            Probability estimates
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
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
        dict
            Dictionary containing accuracy, precision, recall, and F1 score
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        y_pred = self.predict(X_test)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),

        }

    def save_classifier(self, file_path=None, name_suffix=None):
        """
        Save the trained model to a file.

        Parameters:
        -----------
        file_path : str or Path
            Path to save the model
        """
        # Compute the default file path if none is provided
        if file_path is None:
            file_path = f"{DATA_ROOT}/project2_data/{type(self).__name__}_{name_suffix}"
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.classifier, f)

    @classmethod
    def load_model(cls, name, file_path=None):
        """
        Load a saved model from file.

        Parameters:
        -----------
        file_path : str or Path
            Path to the saved model

        Returns:
        --------
        Classifier
            Loaded classifier instance
        """

        # Compute the default file path if none is provided
        if file_path is None:
            file_path = f"{DATA_ROOT}/project2_data/{name}"

        file_path = Path(file_path)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Create a new instance without calling __init__
        classifier = cls.__new__(cls)
        classifier.__dict__.update(data)

        return classifier
