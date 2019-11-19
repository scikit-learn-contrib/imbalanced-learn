"""
==========================================================================
Illustration of the sample selection for the different SPIDER algorithms
==========================================================================

This example illustrates the different ways of resampling with SPIDER.

"""

# Authors: Matthew Eding
# License: MIT

from collections import namedtuple
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from imblearn.combine import SPIDER
from matplotlib.patches import Circle
from scipy.stats import mode

print(__doc__)

###############################################################################
# These are helper functions for plotting aspects of the algorithm

Neighborhood = namedtuple('Neighborhood', 'radius, neighbors')


def plot_X(X, ax, **kwargs):
    ax.scatter(X[:, 0], X[:, 1], **kwargs)


def correct(nn, y_fit, X, y, additional=False):
    n_neighbors = nn.n_neighbors
    if additional:
        n_neighbors += 2
    nn_idxs = nn.kneighbors(X, n_neighbors, return_distance=False)[:, 1:]
    y_pred, _ = mode(y_fit[nn_idxs], axis=1)
    return (y == y_pred.ravel())


def get_neighborhoods(spider, X_fit, y_fit, X_flagged, y_flagged, idx):
    point = X_flagged[idx]

    additional = (spider.kind == 'strong')
    if correct(spider.nn_, y_fit, point[np.newaxis],
               y_flagged[idx][np.newaxis], additional=additional):
        additional = False

    idxs_k = spider._locate_neighbors(point[np.newaxis])
    neighbors_k = X_fit[idxs_k].squeeze()
    farthest_k = neighbors_k[-1]
    radius_k = np.linalg.norm(point - farthest_k)
    neighborhood_k = Neighborhood(radius_k, neighbors_k)

    idxs_k2 = spider._locate_neighbors(point[np.newaxis], additional=True)
    neighbors_k2 = X_fit[idxs_k2].squeeze()
    farthest_k2 = neighbors_k2[-1]
    radius_k2 = np.linalg.norm(point - farthest_k2)
    neighborhood_k2 = Neighborhood(radius_k2, neighbors_k2)

    return neighborhood_k, neighborhood_k2, point, additional


def draw_neighborhoods(spider, neighborhood_k, neighborhood_k2, point,
                       additional, ax, outer=True, alpha=0.5):
    PartialCircle = partial(Circle, facecolor='none', edgecolor='black',
                            alpha=alpha)

    circle_k = PartialCircle(point, neighborhood_k.radius, linestyle='-')

    circle_k2 = PartialCircle(point, neighborhood_k2.radius,
                              linestyle=('-' if additional else '--'))

    if not additional:
        ax.add_patch(circle_k)

    if (spider.kind == 'strong') and outer:
        ax.add_patch(circle_k2)


def draw_amplification(X_flagged, point, neighbors, ax):
    for neigh in neighbors:
        arr = np.vstack([point, neigh])
        xs, ys = np.split(arr, 2, axis=1)
        linestyle = 'solid' if neigh in X_flagged else 'dotted'
        ax.plot(xs, ys, color='black', linestyle=linestyle)


def plot_spider(kind, X, y):
    if kind == 'strong':
        _, axes = plt.subplots(2, 1, figsize=(12, 16))
    else:
        _, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes = np.atleast_1d(axes)

    spider = SPIDER(kind=kind)
    spider.fit_resample(X, y)

    is_safe = correct(spider.nn_, y, X, y)
    is_minor = (y == 1)

    X_major = X[~is_minor]
    X_minor = X[is_minor]
    X_noise = X[~is_safe]

    X_minor_noise = X[is_minor & ~is_safe]
    y_minor_noise = y[is_minor & ~is_safe]
    X_major_safe = X[~is_minor & is_safe]
    X_minor_safe = X[is_minor & is_safe]
    y_minor_safe = y[is_minor & is_safe]

    partial_neighborhoods = partial(get_neighborhoods, spider, X, y)
    partial_amplification = partial(draw_amplification, X_major_safe)
    partial_draw_neighborhoods = partial(draw_neighborhoods, spider)

    size = 500
    for axis in axes:
        plot_X(X_minor, ax=axis, label='Minority class', s=size, marker='_')
        plot_X(X_major, ax=axis, label='Minority class', s=size, marker='+')

        #: Overlay ring around noisy samples for both classes
        plot_X(X_noise, ax=axis, label='Noisy Sample', s=size, marker='o',
               facecolors='none', edgecolors='black')

    #: Neighborhoods for Noisy Minority Samples
    for idx in range(len(X_minor_noise)):
        neighborhoods = partial_neighborhoods(X_minor_noise, y_minor_noise,
                                              idx=idx)
        partial_draw_neighborhoods(*neighborhoods, ax=axes[0],
                                   outer=(spider.kind == 'strong'))
        neigh_k, neigh_k2, point, additional = neighborhoods
        neighbors = neigh_k2.neighbors if additional else neigh_k.neighbors
        partial_amplification(point, neighbors, ax=axes[0])

    axes[0].axis('equal')
    axes[0].legend(markerscale=0.5)
    axes[0].set_title(f'SPIDER-{spider.kind.title()}')

    #: Neighborhoods for Safe Minority Samples (kind='strong' only)
    if spider.kind == 'strong':
        for idx in range(len(X_minor_safe)):
            neighborhoods = partial_neighborhoods(X_minor_safe, y_minor_safe,
                                                  idx=idx)
            neigh_k, _, point, additional = neighborhoods
            neighbors = neigh_k.neighbors
            draw_flag = np.any(np.isin(neighbors, X_major_safe))

            alpha = 0.5 if draw_flag else 0.1
            partial_draw_neighborhoods(*neighborhoods[:-1], additional=False,
                                       ax=axes[1], outer=False, alpha=alpha)

            if draw_flag:
                partial_amplification(point, neighbors, ax=axes[1])

            axes[1].axis('equal')
            axes[1].legend(markerscale=0.5)
            axes[1].set_title(f'SPIDER-{spider.kind.title()}')


###############################################################################
# We can start by generating some data to later illustrate the principle of
# each SPIDER heuritic rules.

X = np.array([
    [-11.83, -6.81],
    [-11.72, -2.34],
    [-11.43, -5.85],
    [-10.66, -4.33],
    [-9.64, -7.05],
    [-8.39, -4.41],
    [-8.07, -5.66],
    [-7.28, 0.91],
    [-7.24, -2.41],
    [-6.13, -4.81],
    [-5.92, -6.81],
    [-4., -1.81],
    [-3.96, 2.67],
    [-3.74, -7.31],
    [-2.96, 4.69],
    [-1.56, -2.33],
    [-1.02, -4.57],
    [0.46, 4.07],
    [1.2, -1.53],
    [1.32, 0.41],
    [1.56, -5.19],
    [2.52, 5.89],
    [3.03, -4.15],
    [4., -0.59],
    [4.4, 2.07],
    [4.41, -7.45],
    [4.45, -4.12],
    [5.13, -6.28],
    [5.4, -5],
    [6.26, 4.65],
    [7.02, -6.22],
    [7.5, -0.11],
    [8.1, -2.05],
    [8.42, 2.47],
    [9.62, 3.87],
    [10.54, -4.47],
    [11.42, 0.01]
])

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
              0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0])


###############################################################################
# SPIDER-Weak / SPIDER-Relabel
###############################################################################

###############################################################################
# Both SPIDER-Weak and SPIDER-Relabel start by labeling whether samples are
# 'safe' or 'noisy' by looking at each point's 3-NN and seeing if it would be
# classified correctly using KNN classification. For each minority-noisy
# sample, we amplify it by the number of majority-safe samples in its 3-NN. In
# the diagram below, the amplification amount is indicated by the number of
# solid lines for a given minority-noisy sample's neighborhood.
#
# We can observe that the leftmost minority-noisy sample will be duplicated 3
# times, the middle one 1 time, and the rightmost one will not be amplified.
#
# Then if SPIDER-Weak, every majority-noisy sample is removed from the dataset.
# Othewise if SPIDER-Relabel, we relabel their class to be the minority class
# instead. These would be the samples indicated by a circled plus-sign.

plot_spider('weak', X, y)

###############################################################################
# SPIDER-Strong
###############################################################################

###############################################################################
# SPIDER-Strong still uses 3-NN to classify samples as 'safe' or 'noisy' as the
# first step. However for the amplification step, each minority-noisy sample
# looks at its 5-NN, and if the larger neighborhood still misclassifies the
# sample, the 5-NN is used to amplify. Otherwise if the sample is correctly
# classified with 5-NN, the regular 3-NN is used to amplify.
#
# In the diagram below, we can see that the left/rightmost minority-noisy
# samples are misclassified using 5-NN and will be amplified by 5 and 1
# respectively. The middle minority-noisy sample is classified correctly by
# using 5-NN, so amplification will be done using 3-NN.
#
# Next for each minority-safe sample, the amplification process is applied
# using 3-NN. In the lower subplot, all but one of these samples will not be
# amplified since they do not have majority-safe samples in their
# neighborhoods. The one minority-safe sample to be amplified is indicated in a
# darker neighborhood with lines.

plot_spider('strong', X, y)

plt.show()
