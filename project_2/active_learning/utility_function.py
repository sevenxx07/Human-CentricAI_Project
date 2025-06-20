import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


class UtilityFunction:
    def __init__(self, name):
        self.name = name

    def apply(self, clf, X_pool):
        if self.name == "lc":
            probs = clf.predict_proba(X_pool)
            return 1 - np.max(probs, axis=1)
        elif self.name == "margin":
            probs = clf.predict_proba(X_pool)
            part = np.partition(-probs, 1, axis=1)
            margin = - (part[:, 0] - (-part[:, 1]))
            return margin
        elif self.name == "entropy":
            probs = clf.predict_proba(X_pool)
            log_probs = np.log(probs + 1e-10)
            entropy = -np.sum(probs * log_probs, axis=1)
            return entropy
        elif self.name == "density":
            sim_matrix = cosine_similarity(X_pool)
            density = np.mean(sim_matrix, axis=1)
            return density
        elif self.name == "cluster":
            kmeans = KMeans(n_clusters=10, n_init='auto', random_state=42)
            kmeans.fit(X_pool)
            centers = kmeans.cluster_centers_
            sim_to_center = cosine_similarity(X_pool, centers)
            return np.max(sim_to_center, axis=1)
        else:
            raise ValueError("Unknown utility function")

    def apply_hybrid(self, type, info_scores, rep_scores, alpha=0.5):
        if type == 'sum':
            return info_scores + alpha * rep_scores
        elif type == 'product':
            return info_scores * rep_scores
        else:
            raise ValueError("Unknown hybrid method: choose 'sum' or 'product'")