"""
==================================
Repeated Edited nearest-neighbours
==================================

An illustration of the repeated edited nearest-neighbours method.

"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

from imblearn.under_sampling import (EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours)

print(__doc__)

sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()


# Generate the dataset
X, y = make_classification(n_classes=2, class_sep=1.25, weights=[0.3, 0.7],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=5, n_clusters_per_class=1,
                           n_samples=5000, random_state=10)

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X)

# Three subplots, unpack the axes array immediately
f, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=.5,
            edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=.5,
            edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
ax1.set_title('Original set')

# Apply the ENN
print('ENN')
enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_sample(X, y)
X_res_vis = pca.transform(X_resampled)

ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
            label="Class #0", alpha=.5, edgecolor=almost_black,
            facecolor=palette[0], linewidth=0.15)
ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
            label="Class #1", alpha=.5, edgecolor=almost_black,
            facecolor=palette[2], linewidth=0.15)
ax2.set_title('ENN')

# Apply the RENN
print('RENN')
renn = RepeatedEditedNearestNeighbours()
X_resampled, y_resampled = renn.fit_sample(X, y)
X_res_vis = pca.transform(X_resampled)

ax3.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
            label="Class #0", alpha=.5, edgecolor=almost_black,
            facecolor=palette[0], linewidth=0.15)
ax3.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
            label="Class #1", alpha=.5, edgecolor=almost_black,
            facecolor=palette[2], linewidth=0.15)
ax3.set_title('RENN')

plt.tight_layout()
plt.show()
