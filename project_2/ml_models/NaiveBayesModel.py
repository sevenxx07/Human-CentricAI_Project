import numpy as np
from overrides import override
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from typing import Literal
from project_2.ml_models.ClassifierWrapper import ClassifierWrapper


class NaiveBayesModel(ClassifierWrapper):
    """
    Naive Bayes Classifier implementation that inherits from the base Classifier.
    Supports different variants of Naive Bayes.
    """

    def __init__(self,
                 variant: Literal['gaussian', 'multinomial', 'bernoulli'] = 'gaussian',
                 alpha: float = 1.0,
                 fit_prior: bool = True,
                 random_state: int = None,
                 verbose: bool = True):
        """
        Initialize the Naive Bayes classifier with specified parameters.

        Parameters:
        -----------
        variant : str, default='gaussian'
            Type of Naive Bayes classifier ('gaussian', 'multinomial', 'bernoulli')
        alpha : float, default=1.0
            Additive (Laplace/Lidstone) smoothing parameter
        fit_prior : bool, default=True
            Whether to learn class prior probabilities
        random_state : int, optional
            Random state for reproducibility (only affects BernoulliNB)
        verbose : bool, default=False
            Enable verbose output
        """
        super().__init__(random_state=random_state, verbose=verbose)
        self.variant = variant
        self.alpha = alpha
        self.fit_prior = fit_prior

        # Initialize the appropriate Naive Bayes variant
        if variant == 'gaussian':
            self.classifier = GaussianNB()
        elif variant == 'multinomial':
            self.classifier = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        elif variant == 'bernoulli':
            self.classifier = BernoulliNB(alpha=alpha, fit_prior=fit_prior)
        else:
            raise ValueError(f"Unknown Naive Bayes variant: {variant}. "
                             "Choose from 'gaussian', 'multinomial', 'bernoulli'")

    @override
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set.

        Parameters:
        -----------
        X_test : array-like
            Test feature vectors
        y_test : array-like
            True labels for the test set

        Returns:
        --------
        dict : Evaluation metrics including accuracy, precision, recall, and F1 score.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        # Convert sparse matrices to dense
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()

        # Convert to numpy arrays
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)
        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)

        # Basic validation
        if X_test.size == 0 or y_test.size == 0:
            raise ValueError("Test data cannot be empty")

        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(f"Sample count mismatch: X_test={X_test.shape[0]}, y_test={y_test.shape[0]}")

        return super().evaluate(X_test, y_test)

    def get_hyperparameters(self):
        """
        Return hyperparameters as a dictionary.

        Returns:
        --------
        dict : Hyperparameters of the Naive Bayes model
        """
        return {
            'variant': self.variant,
            'alpha': self.alpha,
            'fit_prior': self.fit_prior
        }