from sklearn.svm import SVC
from project_2.ML_models.ClassifierWrapper import ClassifierWrapper


class SVMModel(ClassifierWrapper):
    """
    SVM Classifier implementation that inherits from the base Classifier.
    """

    def __init__(self, kernel="linear", C=1.0, random_state=42, verbose=True):
        """
        Initialize the SVM classifier with specified parameters.
        """
        super().__init__(random_state=random_state, verbose=verbose)
        self.kernel = kernel
        self.C = C
        self.classifier = SVC(
            kernel=kernel,
            C=C,
            random_state=random_state,
            verbose=verbose,
            probability=True
        )

    def train(self, X_train, y_train):
        """
        Train the SVM classifier on the provided data.
        """
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        return self
