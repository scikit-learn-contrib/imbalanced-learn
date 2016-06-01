"""
===========================
Instance Hardness Threshold
===========================

An illustration of the instance hardness threshold method.

"""

print(__doc__)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from unbalanced_dataset.under_sampling import InstanceHardnessThreshold

# Generate the dataset
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=10)

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X)

# Two subplots, unpack the axes array immediately
f, axs = plt.subplots(2, 2)

axs = [a for ax in axs for a in ax]
for ax, threshold in zip(axs, [0.0,0.25,0.5,0.75]):
    if threshold == 0.0:
        ax.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
                    edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
        ax.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5,
                    edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
        ax.set_title('Original set')
    else:
        estimator = DecisionTreeClassifier(max_depth=2)
        iht  = InstanceHardnessThreshold(estimator, threshold=threshold)
        X_resampled, y_resampled = iht.fit_transform(X, y)
        X_res_vis = pca.transform(X_resampled)

        ax.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
                    label="Class #0", alpha=.5, edgecolor=almost_black,
                    facecolor=palette[0], linewidth=0.15)
        ax.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
                    label="Class #1", alpha=.5, edgecolor=almost_black,
                    facecolor=palette[2], linewidth=0.15)
        ax.set_title('Instance Hardness Threshold ({})'.format(threshold))

plt.show()
