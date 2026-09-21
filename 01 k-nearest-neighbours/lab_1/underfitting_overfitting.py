import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold

from k_NN import KNNClassifier

X, y = make_moons(n_samples=600, noise=0.35, random_state=0)

def train_val_curves(X, y, ks, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_acc = np.zeros((len(ks), n_splits))
    va_acc = np.zeros((len(ks), n_splits))
    for f, (tr, va) in enumerate(skf.split(X, y)):
        for i, k in enumerate(ks):
            m = KNNClassifier(k=k).fit(X[tr], y[tr])
            tr_acc[i, f] = (m.predict(X[tr]) == y[tr]).mean()
            va_acc[i, f] = (m.predict(X[va]) == y[va]).mean()
    return tr_acc, va_acc

ks = np.arange(1, 101, step=2)  # odd k to avoid ties
tr_acc, va_acc = train_val_curves(X, y, ks)

fig, ax = plt.subplots(figsize=(8, 4.5))
for acc, label, c in [(tr_acc, "train", "tab:blue"), (va_acc, "validation", "tab:orange")]:
    m, s = acc.mean(axis=1), acc.std(axis=1)
    ax.plot(ks, m, label=label, color=c)
    ax.fill_between(ks, m - s, m + s, color=c, alpha=0.2)   # fold-to-fold spread

k_best = ks[va_acc.mean(axis=1).argmax()]
ax.axvline(k_best, ls="--", c="gray")
ax.axvspan(ks.min(), k_best * 0.4, color="red", alpha=0.07, label="overfitting")
ax.axvspan(k_best * 2.5, ks.max(), color="green", alpha=0.07, label="underfitting")
ax.set_xlabel("k  (model complexity increases  →)"); ax.set_ylabel("accuracy")
ax.legend()
plt.suptitle("Overfitting & Underfitting - only ODD k")
plt.show()



# ======================================

gap = tr_acc.mean(axis=1) - va_acc.mean(axis=1)
plt.plot(ks, gap); plt.axhline(0, c="k", lw=0.5)
plt.suptitle("Gap curve between train and validation accuracy - only ODD k")
plt.xlabel("k"); plt.ylabel("train acc - validation acc")
plt.show()

# =====================================
