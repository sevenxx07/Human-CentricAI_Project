from overrides import override
from sklearn.svm import SVC
from project_2.ml_models.ClassifierWrapper import ClassifierWrapper


class SVMModel(ClassifierWrapper):
    """
    SVM Classifier implementation that inherits from the base Classifier.
    """

    def __init__(self, kernel="linear", C=1.0, gamma='scale', max_iter=1000, random_state=42, verbose=True):
        """
        Initialize the SVM classifier with specified parameters.
        """
        super().__init__(random_state=random_state, verbose=verbose)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.classifier = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
            probability=True,
            tol=1e-3       # Relax tolerance for faster convergence
        )

    def get_hyperparameters(self):
        """
        Return hyperparameters as a dictionary.

        Returns:
        --------
        dict : Hyperparameters of the model
        """
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'max_iter': self.max_iter
        }