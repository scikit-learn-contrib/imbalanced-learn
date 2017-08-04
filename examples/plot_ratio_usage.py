"""
============================================================
Usage of the ``ratio`` parameter for the different algorithm
============================================================

This example shows how to use the ``ratio`` parameter in the different
examples. It illustrated the use of passing ``ratio`` as a ``str``, ``dict`` or
a callable.

"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import RandomUnderSampler

print(__doc__)


def plot_pie(y):
    target_stats = Counter(y)
    labels = list(target_stats.keys())
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True,
           autopct='%1.1f%%')
    ax.axis('equal')


###############################################################################
# Creation of an imbalanced data set from a balanced data set
###############################################################################

###############################################################################
# We will show how to use the parameter ``ratio`` when dealing with the
# ``make_imbalance`` function. For this function, this parameter accepts both
# dictionary and callable. When using a dictionary, each key will correspond to
# the class of interest and the corresponding value will be the number of
# samples desired in this class.

iris = load_iris()

print('Information of the original iris data set: \n {}'.format(
    Counter(iris.target)))
plot_pie(iris.target)

ratio = {0: 10, 1: 20, 2: 30}
X, y = make_imbalance(iris.data, iris.target, ratio=ratio)

print('Information of the iris data set after making it'
      ' imbalanced using a dict: \n ratio={} \n y: {}'.format(ratio,
                                                              Counter(y)))
plot_pie(y)

###############################################################################
# You might required more flexibility and require your own heuristic to
# determine the number of samples by class and you can define your own callable
# as follow. In this case we will define a function which will use a float
# multiplier to define the number of samples per class.


def ratio_multiplier(y):
    multiplier = {0: 0.5, 1: 0.7, 2: 0.95}
    target_stats = Counter(y)
    for key, value in target_stats.items():
        target_stats[key] = int(value * multiplier[key])
    return target_stats


X, y = make_imbalance(iris.data, iris.target, ratio=ratio_multiplier)

print('Information of the iris data set after making it'
      ' imbalanced using a callable: \n ratio={} \n y: {}'.format(
          ratio_multiplier, Counter(y)))
plot_pie(y)

###############################################################################
# Using ``ratio`` in resampling algorithm
###############################################################################

###############################################################################
# In all sampling algorithms, ``ratio`` can be used as illustrated earlier. In
# addition, some predefined functions are available and can be executed using a
# ``str`` with the following choices: (i) ``'minority'``: resample the minority
# class; (ii) ``'majority'``: resample the majority class, (iii) ``'not
# minority'``: resample all classes apart of the minority class, (iv)
# ``'all'``: resample all classes, and (v) ``'auto'``: correspond to 'all' with
# for over-sampling methods and 'not minority' for under-sampling methods. The
# classes targeted will be over-sampled or under-sampled to achieve an equal
# number of sample with the majority or minority class.

ratio = 'auto'
X_res, y_res = RandomUnderSampler(ratio=ratio, random_state=0).fit_sample(X, y)

print('Information of the iris data set after balancing using "auto"'
      ' mode:\n ratio={} \n y: {}'.format(ratio, Counter(y_res)))
plot_pie(y_res)

###############################################################################
# However, you can use the dictionary or the callable options as previously
# mentioned.

ratio = {0: 25, 1: 30, 2: 35}
X_res, y_res = RandomUnderSampler(ratio=ratio, random_state=0).fit_sample(X, y)

print('Information of the iris data set after balancing using a dict'
      ' mode:\n ratio={} \n y: {}'.format(ratio, Counter(y_res)))
plot_pie(y_res)


def ratio_multiplier(y):
    multiplier = {1: 0.7, 2: 0.95}
    target_stats = Counter(y)
    for key, value in target_stats.items():
        target_stats[key] = int(value * multiplier[key])
    return target_stats


X_res, y_res = RandomUnderSampler(ratio=ratio, random_state=0).fit_sample(X, y)

print('Information of the iris data set after balancing using a callable'
      ' mode:\n ratio={} \n y: {}'.format(ratio, Counter(y_res)))
plot_pie(y_res)

plt.show()
