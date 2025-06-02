"""Here is how to use the script:
python Logistic_regression.py --vectors tfidf_vectors.npy --save_model tfidf_classifier.pkl
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_vectors(path):
    return np.load(path)


def load_labels(csv_path="cleaned_imdb_reviews.csv"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    labels = df['sentiment'].map({'positive': 1, 'negative': 0}).values
    return labels

class LogRegression:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.clf = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_classifier(self):
        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(self.X, self.y)
        return self.clf

    def evaluate_classifier(self):
        y_pred = self.clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        return acc


    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)


    def load_model(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Logistic Regression Classifier")
    parser.add_argument("--vectors", type=str, required=True, help="Path to .npy vector file")
    parser.add_argument("--labels", type=str, default="cleaned_imdb_reviews.csv", help="Path to CSV with sentiment labels")
    parser.add_argument("--save_model", type=str, default="classifier.pkl", help="Where to save the trained classifier")

    args = parser.parse_args()

    print("Loading vectors and labels...")
    X = load_vectors(args.vectors)
    y = load_labels(args.labels)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training classifier...")
    model = LogisticRegression(X_train, X_test, y_train, y_test)
    model.train_classifier()

    print("📊 Evaluating classifier...")
    acc = model.evaluate_classifier()
    print(f"Accuracy on test set: {acc:.4f}")

    print(f"Saving classifier to: {args.save_model}")
    model.save_model(args.save_model)
