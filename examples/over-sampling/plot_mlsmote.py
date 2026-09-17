"""
====================================
MLSMOTE for imbalanced multilabel data
====================================

This example illustrates :class:`~imblearn.over_sampling.MLSMOTE`, the only
over-sampler in imbalanced-learn supporting multilabel targets. A synthetic
multilabel dataset is created with one rare label. MLSMOTE detects it through
the imbalance ratio per label and generates synthetic samples carrying it.
"""

# Authors: Omar Alaaeldein
# License: MIT
# %%
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_multilabel_classification

from imblearn.over_sampling import MLSMOTE

X, y = make_multilabel_classification(
    n_samples=200, n_features=2, n_classes=3, n_labels=1, random_state=0
)
print(f"Label counts before resampling: {y.sum(axis=0).tolist()}")

sampler = MLSMOTE(random_state=0)
X_res, y_res = sampler.fit_resample(X, y)
print(f"Label counts after resampling: {y_res.sum(axis=0).tolist()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, (X_plot, y_plot, title) in zip(
    axes,
    [(X, y, "Original dataset"), (X_res, y_res, "Resampled with MLSMOTE")],
):
    # color each sample by its first active label for display purposes
    colors = np.argmax(np.hstack([y_plot, np.zeros((y_plot.shape[0], 1))]), axis=1)
    scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=colors, alpha=0.7, edgecolor="k")
    ax.set_title(title)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
fig.legend(*scatter.legend_elements(), title="First active label")
plt.tight_layout()
plt.show()
