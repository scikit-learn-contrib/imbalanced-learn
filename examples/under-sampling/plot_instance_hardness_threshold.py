"""
===========================
Instance Hardness Threshold
===========================

An illustration of the instance hardness threshold method.

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
from sklearn.svm import SVC

from unbalanced_dataset.under_sampling import InstanceHardnessThreshold

# Generate the dataset
X, y = make_classification(n_classes=2, class_sep=1., weights=[0.05, 0.95],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=10)

pca = PCA(n_components=2)
X = pca.fit_transform(X)

# Two subplots, unpack the axes array immediately
f, axs = plt.subplots(2, 2)

axs = [a for ax in axs for a in ax]
for ax, ratio in zip(axs, [0.0, 0.1, 0.3, 0.5]):
    if ratio == 0.0:
        ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5,
                   edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5,
                   edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
        ax.set_title('Original set')
    else:
        estimator = SVC(probability=True)
        iht = InstanceHardnessThreshold(estimator, ratio=ratio)
        X_res, y_res = iht.fit_transform(X, y)
        print np.bincount(y_res)

        ax.scatter(X_res[y_res == 0, 0], X_res[y_res == 0, 1],
                   label="Class #0", alpha=.5, edgecolor=almost_black,
                   facecolor=palette[0], linewidth=0.15)
        ax.scatter(X_res[y_res == 1, 0], X_res[y_res == 1, 1],
                   label="Class #1", alpha=.5, edgecolor=almost_black,
                   facecolor=palette[2], linewidth=0.15)
        ax.set_title('Instance Hardness Threshold ({})'.format(ratio))

plt.show()
