"""
===============
Balance cascade
===============

An illustration of the balance cascade ensemble method.

"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

from unbalanced_dataset.ensemble import BalanceCascade

# Generate the dataset
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=10)

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X)

# Apply Balance Cascade method
bc = BalanceCascade()
X_resampled, y_resampled = bc.fit_sample(X, y)
X_res_vis = []
for X_res in X_resampled:
    X_res_vis.append(pca.transform(X_res))

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
            edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5,
            edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
ax1.set_title('Original set')

ax2.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
            edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
for iy, e in enumerate(X_res_vis):
    ax2.scatter(e[y_resampled[iy] == 1, 0], e[y_resampled[iy] == 1, 1],
                label="Class #1", alpha=0.5, edgecolor=almost_black,
                facecolor=np.random.rand(3,), linewidth=0.15)
ax2.set_title('Balance cascade')

plt.show()
