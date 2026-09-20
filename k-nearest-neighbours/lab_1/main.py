"""Validation of the KNN algorithm on the Iris dataset and compare it against scikit-learn's implementation."""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from k_NN import KNNClassifier

if __name__ == "__main__":

    X, y = load_iris(return_X_y=True)
    X2 = X[:, :2]  # Use only the first two features for visualization

    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=0, stratify=y)

    # Sanity check: our implementation should match scikit-learn
    scratch_knn = KNNClassifier(k=5, p=2).fit(X_train, y_train).predict(X_test)
    sklearn_knn = KNeighborsClassifier(n_neighbors=5, p=2).fit(X_train, y_train).predict(X_test)

    print("Agreement with sklearn: ", (
                scratch_knn == sklearn_knn).mean())  # ties may cause tiny differences in predictions, but should be very close to 1.0

    # Decision boundaries for different k
    fig, axes = plt.subplots(3, 1, figsize=(5, 12))
    xx, yy = np.meshgrid(
        np.linspace(X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5, 200),
        np.linspace(X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    for ax, k in zip(axes, [1, 15, 60]):
        model = KNNClassifier(k=k, p=2).fit(X_train, y_train)
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k")
        ax.set_title(f"k = {k}")

    plt.tight_layout()
    plt.show()