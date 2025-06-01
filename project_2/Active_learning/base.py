import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from project_2.Active_learning.utility_function import UtilityFunction
from project_2.ML_models.Logistic_regression import LogRegression


def load_vectors(path):
    return np.load(path)


def load_labels(csv_path="imdb_dataset.csv"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df['sentiment'].map({'positive': 1, 'negative': 0}).values


# ========== Active Learning Loop ==========

def active_learning_loop(X, y, utility, model, n_initial=10, n_queries=100):
    return None


# ========== Main CLI ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 2: Active Learning")
    parser.add_argument("--vectors", type=str, required=True, help="Path to .npy file with feature vectors")
    parser.add_argument("--labels", type=str, default="imdb_dataset.csv", help="CSV file with sentiment labels")
    parser.add_argument("--strategy", type=str, choices=["lc", "margin", "entropy, density, cluster"], default="lc",
                        help="Active learning utility strategy")
    parser.add_argument("--model", type=str, choices=["logreg, bayes, svm"], default="logreg", help="Model used to training")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries to make")
    parser.add_argument("--init", type=int, default=10, help="Initial labeled samples")
    args = parser.parse_args()

    X = load_vectors(args.vectors)
    y = load_labels(args.labels)

    utility = UtilityFunction(args.strategy)
    #TODO
    if args.model == "logreg":
        model = LogRegression()

    acc = active_learning_loop(X, y, utility, model, n_initial=args.init, n_queries=args.queries)

    # Plot accuracy over time
    plt.plot(range(len(acc)), acc, label=args.strategy.upper())
    plt.xlabel("Number of Queries")
    plt.ylabel("Accuracy")
    plt.title("Active Learning Progress")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("active_learning_accuracy.png")
    plt.show()
