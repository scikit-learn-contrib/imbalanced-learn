"""
===========================
make_imbalance function
===========================

An illustration of the make_imbalance function

"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons

from imblearn.datasets import make_imbalance

print(__doc__)

sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()


# Generate the dataset
X, y = make_moons(n_samples=200, shuffle=True, noise=0.5, random_state=10)

# Two subplots, unpack the axes array immediately
f, axs = plt.subplots(2, 3)

axs = [a for ax in axs for a in ax]

axs[0].scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0",
               alpha=0.5, edgecolor=almost_black, facecolor=palette[0],
               linewidth=0.15)
axs[0].scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1",
               alpha=0.5, edgecolor=almost_black, facecolor=palette[2],
               linewidth=0.15)
axs[0].set_title('Original set')

ratios = [0.9, 0.75, 0.5, 0.25, 0.1]
for i, ratio in enumerate(ratios, start=1):
    ax = axs[i]

    X_, y_ = make_imbalance(X, y, ratio=ratio, min_c_=1)

    ax.scatter(X_[y_ == 0, 0], X_[y_ == 0, 1], label="Class #0",
               alpha=0.5, edgecolor=almost_black, facecolor=palette[0],
               linewidth=0.15)
    ax.scatter(X_[y_ == 1, 0], X_[y_ == 1, 1], label="Class #1",
               alpha=0.5, edgecolor=almost_black, facecolor=palette[2],
               linewidth=0.15)
    ax.set_title('make_imbalance ratio ({})'.format(ratio))

plt.tight_layout()
plt.show()
