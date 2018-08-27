"""
==============================================
Illustration of the definition of a Tomek link
==============================================

This example illustrates what is a Tomek link.

"""

import matplotlib.pyplot as plt
import numpy as np

from imblearn.under_sampling import TomekLinks

print(__doc__)

rng = np.random.RandomState(18)

###############################################################################
# This function allows to make nice plotting


def make_plot_despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([0., 3.5])
    ax.set_ylim([0., 3.5])
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.legend()


###############################################################################
# Generate some data with one Tomek link

# minority class
X_minority = np.transpose([[1.1, 1.3, 1.15, 0.8, 0.55, 2.1],
                           [1., 1.5, 1.7, 2.5, 0.55, 1.9]])
# majority class
X_majority = np.transpose([[2.1, 2.12, 2.13, 2.14, 2.2, 2.3, 2.5, 2.45],
                           [1.5, 2.1, 2.7, 0.9, 1.0, 1.4, 2.4, 2.9]])

###############################################################################
# In the figure above, the samples highlighted in green form a Tomek link since
# they are of different classes and are nearest neighbours of each other.

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X_minority[:, 0], X_minority[:, 1],
           label='Minority class', s=200, marker='_')
ax.scatter(X_majority[:, 0], X_majority[:, 1],
           label='Majority class', s=200, marker='+')

# highlight the samples of interest
ax.scatter([X_minority[-1, 0], X_majority[1, 0]],
           [X_minority[-1, 1], X_majority[1, 1]],
           label='Tomek link', s=200, alpha=0.3)
ax.set_title('Illustration of a Tomek link')
make_plot_despine(ax)
fig.tight_layout()

###############################################################################
# We can run the ``TomekLinks`` sampling to remove the corresponding
# samples. If ``sampling_strategy='auto'`` only the sample from the majority
# class will be removed. If ``sampling_strategy='all'`` both samples will be
# removed.

sampler = TomekLinks()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax_arr = (ax1, ax2)
title_arr = ('Removing only majority samples',
             'Removing all samples')
for ax, title, sampler in zip(ax_arr,
                              title_arr,
                              [TomekLinks(sampling_strategy='auto'),
                               TomekLinks(sampling_strategy='all')]):
    X_res, y_res = sampler.fit_resample(np.vstack((X_minority, X_majority)),
                                        np.array([0] * X_minority.shape[0] +
                                                 [1] * X_majority.shape[0]))
    ax.scatter(X_res[y_res == 0][:, 0], X_res[y_res == 0][:, 1],
               label='Minority class', s=200, marker='_')
    ax.scatter(X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1],
               label='Majority class', s=200, marker='+')

    # highlight the samples of interest
    ax.scatter([X_minority[-1, 0], X_majority[1, 0]],
               [X_minority[-1, 1], X_majority[1, 1]],
               label='Tomek link', s=200, alpha=0.3)

    ax.set_title(title)
    make_plot_despine(ax)
fig.tight_layout()

plt.show()
