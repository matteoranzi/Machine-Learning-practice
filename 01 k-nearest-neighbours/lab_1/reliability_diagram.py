import numpy as np

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from k_NN import KNNClassifier

X, y = make_moons(n_samples=1000, noise=0.35, random_state=0)   # noisy, overlapping classes
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

plt.figure(figsize=(6, 5))
for k in [1, 5, 25, 100]:
    model = KNNClassifier(k=k).fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10)
    ll = log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6))
    plt.plot(mean_pred, frac_pos, "o-", label=f"k={k} (logloss {ll:.2f}, Brier {brier_score_loss(y_te, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
plt.xlabel("predicted probability"); plt.ylabel("observed frequency")
plt.legend(); plt.show()