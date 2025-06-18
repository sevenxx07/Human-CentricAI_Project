# sparse_decision_tree.py

from gosdt import GOSDTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from palmerpenguins import load_penguins
import pandas as pd
import graphviz

class SparseDecisionTree:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self._load_data()
        self.model = None

    def _load_data(self):
        data = load_penguins()
        data = data.dropna()
        self.X = data.drop(columns=["species"])
        self.y = data["species"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def train(self, alpha=0.0, depth_budget=5, time_limit=10):
        self.model = GOSDTClassifier(regularization=alpha, time_limit=time_limit, depth_budget=depth_budget, verbose=False)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        if self.model is None:
            raise RuntimeError("You need to train the model before evaluation.")
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        leaves = self.model.model["solution"]["leaf_count"]
        return acc, leaves

    def export_tree_image(self, filename="tree_visualization"):
        if self.model is None:
            raise RuntimeError("You need to train the model before visualization.")
        dot = self.model.plot()
        graph = graphviz.Source(dot)
        graph.format = 'png'
        out_file = graph.render(filename, cleanup=True)
        return out_file

if __name__ == "__main__":
    tree_model = SparseDecisionTree()
    tree_model.train(alpha=0.1)

    # Evaluate
    acc, leaves = tree_model.evaluate()
    print(f"Accuracy: {acc:.2f}, Leaves: {leaves}")

    # Visualize
    img_path = tree_model.export_tree_image()
    print(f"Tree image saved to: {img_path}")