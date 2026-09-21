import numpy as np
from collections import Counter

class KNNClassifier:
    """"k-NN classifier with Minkowski distance, written from scratch."""

    def __init__(self, k=3, p=2, weights="uniform"):
        self.X_train = None
        self.y_train = None

        self.classes_= None
        self.y_idx = None

        self.k = k
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        """Lazy learner: just memorize the data."""
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

        self.classes_, self.y_idx = np.unique(self.y_train, return_inverse=True)

        return self

    def _distance(self, X):
        """Vectorized pairwise Minkowski distance: shape (n_query, n_train)."""

        # Compute the absolute difference between every query point and every training point
        diff = np.abs(X[:, None, :] - self.X_train[None, :, :]) # shape (n_query, n_train, n_features)
        return (diff ** self.p).sum(axis=2) ** (1 / self.p) # shape (n_query, n_train) where entry [i, j] is the distance between query i and training point j

    def predict(self, X):
        """Predict the class labels for the provided data."""
        X = np.asarray(X, dtype=float)
        D = self._distance(X)

        # Indices of the k smallest distances for each query
        nn_idx = np.argpartition(D, self.k - 1, axis=1)[:, : self.k] # (n_query, k)

        # Sort the k neighbours of each query by their distance (nearest first)
        nn_dist = np.take_along_axis(D, nn_idx, axis=1) # (n_query, k)
        order = np.argsort(nn_dist, axis=1, kind="stable") # "stable" to preserve the order of equal distances
        nn_idx = np.take_along_axis(nn_idx, order, axis=1) # (n_query, k), sorted


        nn_labels = self.y_train[nn_idx]

        # Majority vote per query
        return np.array([Counter(row).most_common(1)[0][0] for row in nn_labels])

    def _neighbours(self, X):
        X = np.asarray(X, dtype=float)
        diff = np.abs(X[:, None, :] - self.X_train[None, :, :])
        D = (diff ** self.p).sum(axis=2) ** (1 / self.p)
        nn_idx = np.argpartition(D, self.k - 1, axis=1)[:, : self.k]
        nn_dist = np.take_along_axis(D, nn_idx, axis=1)
        return nn_idx, nn_dist

    def predict_proba(self, X):
        idx, dist = self._neighbours(X)
        labels = self.y_idx[idx]
        if self.weights == "uniform":
            w = np.ones_like(dist)
        else:
            w = 1.0 / np.maximum(dist, 1e-12) # Avoid division by zero
        C = len(self.classes_)
        P = np.zeros((len(labels), C))
        for c in range(C):
            P[:, c] = (w * (labels == c)).sum(axis=1)
        return P / P.sum(axis=1, keepdims=True)

class KNNRegressor(KNNClassifier):
    def predict(self, X):
        """Predict the target values for the provided data."""
        X = np.asarray(X, dtype=float)
        D = self._distance(X)
        nn_idx = np.argpartition(D, self.k - 1, axis=1)[:, : self.k] # Partition over the distances on axis 1 (train samples) to find the k nearest neighbors
        return self.y_train[nn_idx].mean(axis=1)



