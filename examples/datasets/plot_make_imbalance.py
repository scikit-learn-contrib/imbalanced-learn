"""
============================
Create an imbalanced dataset
============================

An illustration of the :func:`imblearn.datasets.make_imbalance` function to
create an imbalanced dataset from a balanced dataset. We show the ability of
:func:`imblearn.datasets.make_imbalance` of dealing with Pandas DataFrame.

"""

# Authors: Dayvid Oliveira
#          Christos Aridas
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

from imblearn.datasets import make_imbalance

print(__doc__)

# Generate the dataset
X, y = make_moons(n_samples=200, shuffle=True, noise=0.5, random_state=10)
X = pd.DataFrame(X, columns=["feature 1", "feature 2"])

# Two subplots, unpack the axes array immediately
f, axs = plt.subplots(2, 3)

axs = [a for ax in axs for a in ax]

X.plot.scatter(
    x='feature 1', y='feature 2', c=y, ax=axs[0], colormap='viridis',
    colorbar=False
)
axs[0].set_title('Original set')


def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}


multipliers = [0.9, 0.75, 0.5, 0.25, 0.1]
for i, multiplier in enumerate(multipliers, start=1):
    ax = axs[i]

    X_, y_ = make_imbalance(X, y, sampling_strategy=ratio_func,
                            **{"multiplier": multiplier,
                               "minority_class": 1})
    X_.plot.scatter(
        x='feature 1', y='feature 2', c=y_, ax=ax, colormap='viridis',
        colorbar=False
    )
    ax.set_title('Sampling ratio = {}'.format(multiplier))

plt.tight_layout()
plt.show()
