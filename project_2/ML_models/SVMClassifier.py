import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from pathlib import Path
import numpy as np


class SVMClassifier:
    """
    A wrapper class for SVM classification with enhanced functionality and best practices.

    Parameters:
    -----------
    kernel : str, default='linear'
        Specifies the kernel type to be used in the algorithm.
    C : float, default=1.0
        Regularization parameter.
    random_state : int, default=42
        Controls the randomness for reproducible results.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(self, kernel="linear", C=1.0, random_state=42, verbose=False):
        """
        Initialize the SVM classifier with specified parameters.
        """
        self.clf = SVC(
            kernel=kernel,
            C=C,
            random_state=random_state,
            verbose=verbose,
            probability=True  # Enable probability estimates for predict_proba
        )


        self.word_encoder = None
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.is_trained = False

    def train(self, X_train, y_train):
        """
        Train the SVM classifier on the provided data.

        Parameters:
        -----------
        X_train : array-like
            Training feature vectors
        y_train : array-like
            Training target values

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        self.clf.fit(X_train, y_train)
        self.is_trained = True
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
        return self.clf.predict(X)

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
        return self.clf.predict_proba(X)

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
            Dictionary containing accuracy, classification report, and confusion matrix
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        y_pred = self.predict(X_test)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation on the training data.

        Parameters:
        -----------
        X : array-like
            Feature vectors
        y : array-like
            Target values
        cv : int, default=5
            Number of cross-validation folds

        Returns:
        --------
        dict
            Dictionary containing cross-validation results
        """
        scores = cross_val_score(self.clf, X, y, cv=cv)
        return {
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'fold_scores': scores.tolist()
        }

    def save_model(self, file_path="./data/pr2_model/svm"):
        """
        Save the trained model to a file.

        Parameters:
        -----------
        file_path : str or Path
            Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.clf,
                'kernel': self.kernel,
                'C': self.C,
                'random_state': self.random_state
            }, f)

    @classmethod
    def load_model(cls, file_path="./data/pr2_model/svm"):
        """
        Load a saved model from file.

        Parameters:
        -----------
        file_path : str or Path
            Path to the saved model

        Returns:
        --------
        SVMClassifier
            Loaded classifier instance
        """
        file_path = Path(file_path)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        classifier = cls(
            kernel=data['kernel'],
            C=data['C'],
            random_state=data['random_state']
        )
        classifier.clf = data['model']
        classifier.is_trained = True

        return classifier