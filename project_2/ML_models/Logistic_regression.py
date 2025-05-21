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


def load_labels(csv_path="imdb_dataset.csv"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    labels = df['sentiment'].map({'positive': 1, 'negative': 0}).values
    return labels


def train_classifier(X, y):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Logistic Regression Classifier")
    parser.add_argument("--vectors", type=str, required=True, help="Path to .npy vector file")
    parser.add_argument("--labels", type=str, default="imdb_dataset.csv", help="Path to CSV with sentiment labels")
    parser.add_argument("--save_model", type=str, default="classifier.pkl", help="Where to save the trained classifier")

    args = parser.parse_args()

    print("Loading vectors and labels...")
    X = load_vectors(args.vectors)
    y = load_labels(args.labels)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training classifier...")
    clf = train_classifier(X_train, y_train)

    print("📊 Evaluating classifier...")
    acc = evaluate_classifier(clf, X_test, y_test)
    print(f"Accuracy on test set: {acc:.4f}")

    print(f"Saving classifier to: {args.save_model}")
    save_model(clf, args.save_model)
