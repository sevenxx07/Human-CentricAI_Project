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

    def train(self, X_train, y_train):
        """
        Train the Naive Bayes classifier on the provided data.
        """
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        return self

    def partial_fit(self, X, y, classes=None):
        """
        Incremental fit on a batch of samples (for online learning).

        Parameters:
        -----------
        X : array-like
            Feature vectors
        y : array-like
            Target values
        classes : array-like, optional
            List of all possible classes (required for first call)
        """
        if not hasattr(self.classifier, 'partial_fit'):
            raise NotImplementedError(
                f"Partial fit not available for {self.variant} Naive Bayes")

        self.classifier.partial_fit(X, y, classes=classes)
        self.is_trained = True
        return self
