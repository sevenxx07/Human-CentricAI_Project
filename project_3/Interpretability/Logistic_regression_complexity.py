import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from palmerpenguins import load_penguins


class SparseLogisticRegression:
    def __init__(self, alpha=1.0, test_size=0.3, random_state=42):
        """
        :param alpha: regularization strength λ (higher = more sparsity)
        :param test_size: fraction of dataset used as test set
        :param random_state: seed for reproducibility
        """
        self.alpha = alpha
        self.test_size = test_size
        self.random_state = random_state

        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_data(self):
        """Load and preprocess the penguins dataset."""
        df = load_penguins().dropna()
        y = self.label_encoder.fit_transform(df["species"])
        X = pd.get_dummies(df.drop(columns=["species"]))
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        """Train a logistic regression model with L1 regularization."""
        C = 1.0 / self.alpha if self.alpha != 0 else 1e12  # prevent division by zero
        self.model = LogisticRegression(
            penalty="l1",
            C=C,
            solver="liblinear",
            multi_class="ovr",
            max_iter=2000
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Return test accuracy and number of non-zero features used."""
        if self.model is None:
            raise RuntimeError("Model is not trained yet.")

        acc = self.model.score(self.X_test, self.y_test)
        nonzero = np.sum(np.any(self.model.coef_ != 0, axis=0))
        return acc, nonzero

    def print_coefficients(self):
        """Print non-zero coefficients per class."""
        if self.model is None:
            raise RuntimeError("Train the model first.")

        class_names = self.label_encoder.inverse_transform(np.arange(len(self.model.classes_)))
        print("\nMODEL COEFFICIENTS (non-zero only):")
        for i, class_name in enumerate(class_names):
            print(f"\nClass: {class_name}")
            for coef, fname in zip(self.model.coef_[i], self.feature_names):
                if coef != 0:
                    print(f"  {fname}: {coef:.4f}")
    
#----------- Added by S for GUI purposes ----
    def get_nonzero_coefficients(self):
        if self.model is None:
            raise RuntimeError("Train the model first.")
        
        class_names = self.label_encoder.inverse_transform(np.arange(len(self.model.classes_)))
        class_coeffs = {}
        for i, class_name in enumerate(class_names):
            feature_dict = {}
            for coef, fname in zip(self.model.coef_[i], self.feature_names):
                if coef != 0:
                    feature_dict[fname] = round(coef,4)
            class_coeffs[class_name] = feature_dict
        return class_coeffs
    
    def get_used_and_unused_features(self):
        if self.model is None:
            raise RuntimeError("Train the model first.")
        
        used_mask = np.any(self.model.coef_ != 0, axis=0)
        used = [fname for fname, used_flag in zip(self.feature_names, used_mask) if used_flag]
        unused = [fname for fname, used_flag in zip(self.feature_names, used_mask) if not used_flag]
        return used, unused
#----------------------------------------

    def run_pipeline(self):
        self.load_data()
        self.train()
        acc, nonzero = self.evaluate()
        print(f"Test Accuracy: {acc:.3f}")
        print(f"Used Features: {nonzero} / {len(self.feature_names)}")
        self.print_coefficients()
        return acc, nonzero # Added by S

if __name__ == "__main__":
    model = SparseLogisticRegression(alpha=30)
    model.run_pipeline()
    #alphas = [0.1, 1, 10, 30, 50, 70, 100]
    #features_used = []
    #for a in alphas:
    #    model = SparseLogisticRegression(alpha=a)
    #    model.load_data()
    #    model.train()
    #    _, nz = model.evaluate()
    #    features_used.append(nz)

    #plt.plot(alphas, features_used)
    #plt.xlabel("Alpha (λ)")
    #plt.ylabel("Used Features")
    #plt.xscale("log")
    #plt.title("Feature Sparsity vs Regularization")
    #plt.grid()
    #plt.show()


