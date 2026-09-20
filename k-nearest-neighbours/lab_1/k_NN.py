import numpy as np
from collections import Counter

class KNNClassifier:
    """"k-NN classifier with Minkowski distance, written from scratch."""

    def __init__(self, k=3, p=2):
        self.X_train = None
        self.y_train = None

        self.k = k
        self.p = p

    def fit(self, X, y):
        """Lazy learner: just memorize the data."""
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
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
        nn_idx = np.argpartition(D, self.k - 1, axis=1)[:, : self.k]
        nn_labels = self.y_train[nn_idx]

        # Majority vote per query
        return np.array([Counter(row).most_common(1)[0][0] for row in nn_labels])


class KNNRegressor(KNNClassifier):
    def predict(self, X):
        """Predict the target values for the provided data."""
        X = np.asarray(X, dtype=float)
        D = self._distance(X)
        nn_idx = np.argpartition(D, self.k - 1, axis=1) # Partition over the distances on axis 1 (train samples) to find the k nearest neighbors
        return self.y_train[nn_idx].mean(axis=1)



