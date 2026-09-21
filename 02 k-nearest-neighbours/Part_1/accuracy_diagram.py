import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from k_NN import KNNClassifier

# Uses the KNNClassifier from the predict_proba section (it has a p parameter)

def cv_accuracy(X, y, k, p, scale, n_splits=5, seed=0):
    """Return per-fold test accuracies for one (k, p, scale) configuration."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)          # fit on the training fold only
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        model = KNNClassifier(k=k, p=p).fit(Xtr, y[tr])
        accs.append((model.predict(Xte) == y[te]).mean())
    return np.array(accs)

ps = [1, 2, 3, 5, 10, 20]
datasets = {"Iris": load_iris(return_X_y=True), "Wine": load_wine(return_X_y=True)}

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for ax, (name, (X, y)) in zip(axes, datasets.items()):
    for scale, style in [(False, "o--"), (True, "s-")]:
        res = [cv_accuracy(X, y, k=5, p=p, scale=scale) for p in ps]
        mean = np.array([r.mean() for r in res])
        std = np.array([r.std() for r in res])
        label = "standardized" if scale else "raw"
        ax.errorbar(ps, mean, yerr=std, fmt=style, capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xticks(ps); ax.set_xticklabels(ps)
    ax.set_xlabel("Minkowski p"); ax.set_ylabel("CV accuracy")
    ax.set_title(f"{name} (k=5)"); ax.legend()
plt.suptitle("k-NN accuracy on Iris and Wine datasets for different Minkowski p values, with and without feature standardization", fontsize=14)
plt.tight_layout(); plt.show()
