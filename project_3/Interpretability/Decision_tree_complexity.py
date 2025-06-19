# sparse_decision_tree.py

import graphviz
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from gosdt import ThresholdGuessBinarizer, GOSDTClassifier
from sklearn.metrics import accuracy_score

class SparseDecisionTree:
    def __init__(self, alpha=0.01, depth_budget=6, time_limit=60, verbose=True):
        self.alpha = alpha
        self.depth_budget = depth_budget
        self.time_limit = time_limit
        self.verbose = verbose

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_bin = None
        self.X_test_bin = None
        self.ref_labels = None
        self.clf = None

    def load_data(self, test_size=0.2, random_state=42):
        """Load and preprocess the Palmer Penguins dataset."""
        df = load_penguins()
        df = df.dropna()
        y = df["species"]
        X = df.drop(columns=["species"])
        X = pd.get_dummies(X)  # Encode categorical variables
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def guess_thresholds(self):
        """Binarize continuous features using gradient boosting thresholds."""
        enc = ThresholdGuessBinarizer(n_estimators=40, max_depth=1, random_state=42)
        enc.set_output(transform="pandas")
        self.X_train_bin = enc.fit_transform(self.X_train, self.y_train)
        self.X_test_bin = enc.transform(self.X_test)

    def get_reference_labels(self):
        """Generate reference predictions from a black-box model."""
        ref_model = GradientBoostingClassifier(n_estimators=40, max_depth=1, random_state=42)
        ref_model.fit(self.X_train_bin, self.y_train)
        self.ref_labels = ref_model.predict(self.X_train_bin)

    def train(self):
        """Train the GOSDT sparse decision tree."""
        self.clf = GOSDTClassifier(
            regularization=self.alpha,
            similar_support=False,
            time_limit=self.time_limit,
            depth_budget=self.depth_budget,
            verbose=self.verbose
        )
        self.clf.fit(self.X_train_bin, self.y_train, y_ref=self.ref_labels)

    def evaluate(self):
        """Return accuracy on train and test data."""
        train_acc = self.clf.score(self.X_train_bin, self.y_train)
        test_acc = self.clf.score(self.X_test_bin, self.y_test)
        return train_acc, test_acc

    def run_pipeline(self):
        """Complete training pipeline."""
        self.load_data()
        self.guess_thresholds()
        self.get_reference_labels()
        self.train()
        return self.evaluate()

    def export_tree_image(self, filename="tree_visualization"):
        """Export the trained tree as a PNG image using Graphviz."""
        if self.clf is None:
            raise RuntimeError("You need to train the model before visualization.")

        dot = self.clf.plot()  # get dot-format string from GOSDT
        graph = graphviz.Source(dot)
        graph.format = 'png'
        out_file = graph.render(filename, cleanup=True)
        return out_file

if __name__ == "__main__":
    tree_model = SparseDecisionTree(alpha=0.05)
    train_acc, test_acc = tree_model.run_pipeline()

    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")

    # Visualize
    img_path = tree_model.export_tree_image()
    print(f"Tree image saved to: {img_path}")
