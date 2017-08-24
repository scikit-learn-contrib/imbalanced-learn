"""
=================
Cluster centroids
=================

An illustration of the cluster centroids method.

"""

# Authors: Fernando Nogueira
#          Christos Aridas
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

from imblearn.under_sampling import ClusterCentroids

print(__doc__)

# Generate the dataset
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=50, random_state=10)

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X)

# Apply Cluster Centroids
cc = ClusterCentroids()
X_resampled, y_resampled = cc.fit_sample(X, y)
X_res_vis_soft = pca.transform(X_resampled)

# Use hard voting instead of soft voting
cc = ClusterCentroids(voting='hard')
X_resampled, y_resampled = cc.fit_sample(X, y)
X_res_vis_hard = pca.transform(X_resampled)

# Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0",
                 alpha=0.5)
c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1",
                 alpha=0.5)
ax1.set_title('Original set')

ax2.scatter(X_res_vis_soft[y_resampled == 0, 0],
            X_res_vis_soft[y_resampled == 0, 1],
            label="Class #0", alpha=.5)
ax2.scatter(X_res_vis_soft[y_resampled == 1, 0],
            X_res_vis_soft[y_resampled == 1, 1],
            label="Class #1", alpha=.5)
c2 = ax2.scatter(X_vis[y == 1, 0],
                 X_vis[y == 1, 1], label="Original #1",
                 alpha=0.2)
ax2.set_title('Cluster centroids with soft voting')

ax3.scatter(X_res_vis_hard[y_resampled == 0, 0],
            X_res_vis_hard[y_resampled == 0, 1],
            label="Class #0", alpha=.5)
ax3.scatter(X_res_vis_hard[y_resampled == 1, 0],
            X_res_vis_hard[y_resampled == 1, 1],
            label="Class #1", alpha=.5)
ax3.scatter(X_vis[y == 1, 0],
            X_vis[y == 1, 1],
            alpha=0.2)
ax3.set_title('Cluster centroids with hard voting')

# make nice plotting
for ax in (ax1, ax2, ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-6, 8])
    ax.set_ylim([-6, 6])

plt.figlegend((c0, c1), ('Class #0', 'Class #1', 'Original Class #1'),
              loc='lower center',
              ncol=3, labelspacing=0.)
plt.tight_layout(pad=3)
plt.show()
