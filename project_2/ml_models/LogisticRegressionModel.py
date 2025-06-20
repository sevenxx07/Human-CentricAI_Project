from sklearn.linear_model import LogisticRegression

from project_2.ml_models.ClassifierWrapper import ClassifierWrapper


class LogisticRegressionModel(ClassifierWrapper):
    """
    Logistic Regression Classifier implementation that inherits from the base Classifier.
    """

    def __init__(self, penalty='l2', C=1.0, solver='lbfgs', max_iter=100,
                 random_state=42, verbose=True):
        """
        Initialize the Logistic Regression classifier with specified parameters.
        """
        super().__init__(random_state=random_state, verbose=verbose)
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.classifier = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose
        )

    def predict_log_proba(self, X):
        """
        Return log probability estimates for the test data (specific to Logistic Regression).

        Parameters:
        -----------
        X : array-like
            Feature vectors to predict

        Returns:
        --------
        log_probabilities : array
            Log probability estimates
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        return self.classifier.predict_log_proba(X)

    def get_hyperparameters(self):
        """
        Return hyperparameters as a dictionary.

        Returns:
        --------
        dict : Hyperparameters of the model
        """
        return {
            'penalty': self.penalty,
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter
        }