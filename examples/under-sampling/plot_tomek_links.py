"""
===========
Tomek links
===========

An illustration of the Tomek links method.

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from imblearn.under_sampling import TomekLinks

print(__doc__)


# create a synthetic dataset
X, y = make_blobs(n_samples=500, centers=2, n_features=2,
                  random_state=0, center_box=(-5.0, 5.0))

# remove Tomek links
tl = TomekLinks(return_indices=True)
X_resampled, y_resampled, idx_resampled = tl.fit_sample(X, y)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

idx_class_0 = np.flatnonzero(y_resampled == 0)
idx_class_1 = np.flatnonzero(y_resampled == 1)
idx_samples_removed = np.setdiff1d(np.flatnonzero(y == 1),
                                   np.union1d(idx_class_0, idx_class_1))

plt.scatter(X[idx_class_0, 0], X[idx_class_0, 1],
            c='g', alpha=.8, label='Class #0')
plt.scatter(X[idx_class_1, 0], X[idx_class_1, 1],
            c='b', alpha=.8, label='Class #1')
plt.scatter(X[idx_samples_removed, 0], X[idx_samples_removed, 1],
            c='r', alpha=.8, label='Removed samples')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))

plt.title('Under-sampling removing Tomek links')
plt.legend()

plt.show()
