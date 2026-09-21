from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold

X, y = make_moons(n_samples=600, noise=0.35, random_state=0)


fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, k in zip(axes, [1, 30]):
    sizes, tr_s, va_s = learning_curve(
        KNeighborsClassifier(n_neighbors=k), X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(5, shuffle=True, random_state=0))
    for s, lab in [(tr_s, "train"), (va_s, "validation")]:
        ax.plot(sizes, s.mean(axis=1), "o-", label=lab)
        ax.fill_between(sizes, s.mean(1) - s.std(1), s.mean(1) + s.std(1), alpha=0.2)
    ax.set_title(f"k={k}"); ax.set_xlabel("training set size"); ax.legend()
axes[0].set_ylabel("accuracy")
# plt.tight_layout()
plt.suptitle("Learning curves of k-NN classifier on moons dataset", fontsize=16)
plt.show()