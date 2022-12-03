"""
=================================
Bagging classifiers using sampler
=================================

In this example, we show how
:class:`~imblearn.ensemble.BalancedBaggingClassifier` can be used to create a
large variety of classifiers by giving different samplers.

We will give several examples that have been published in the passed year.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

# %%
print(__doc__)

# %% [markdown]
# Generate an imbalanced dataset
# ------------------------------
#
# For this example, we will create a synthetic dataset using the function
# :func:`~sklearn.datasets.make_classification`. The problem will be a toy
# classification problem with a ratio of 1:9 between the two classes.

# %%
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=10_000,
    n_features=10,
    weights=[0.1, 0.9],
    class_sep=0.5,
    random_state=0,
)

# %%
import pandas as pd

pd.Series(y).value_counts(normalize=True)

# %% [markdown]
# In the following sections, we will show a couple of algorithms that have
# been proposed over the years. We intend to illustrate how one can reuse the
# :class:`~imblearn.ensemble.BalancedBaggingClassifier` by passing different
# sampler.

from sklearn.ensemble import BaggingClassifier

# %%
from sklearn.model_selection import cross_validate

ebb = BaggingClassifier()
cv_results = cross_validate(ebb, X, y, scoring="balanced_accuracy")

print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")

# %% [markdown]
# Exactly Balanced Bagging and Over-Bagging
# -----------------------------------------
#
# The :class:`~imblearn.ensemble.BalancedBaggingClassifier` can use in
# conjunction with a :class:`~imblearn.under_sampling.RandomUnderSampler` or
# :class:`~imblearn.over_sampling.RandomOverSampler`. These methods are
# referred as Exactly Balanced Bagging and Over-Bagging, respectively and have
# been proposed first in [1]_.

# %%
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler

# Exactly Balanced Bagging
ebb = BalancedBaggingClassifier(sampler=RandomUnderSampler())
cv_results = cross_validate(ebb, X, y, scoring="balanced_accuracy")

print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")

# %%
from imblearn.over_sampling import RandomOverSampler

# Over-bagging
over_bagging = BalancedBaggingClassifier(sampler=RandomOverSampler())
cv_results = cross_validate(over_bagging, X, y, scoring="balanced_accuracy")

print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")

# %% [markdown]
# SMOTE-Bagging
# -------------
#
# Instead of using a :class:`~imblearn.over_sampling.RandomOverSampler` that
# make a bootstrap, an alternative is to use
# :class:`~imblearn.over_sampling.SMOTE` as an over-sampler. This is known as
# SMOTE-Bagging [2]_.

# %%
from imblearn.over_sampling import SMOTE

# SMOTE-Bagging
smote_bagging = BalancedBaggingClassifier(sampler=SMOTE())
cv_results = cross_validate(smote_bagging, X, y, scoring="balanced_accuracy")

print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")

# %% [markdown]
# Roughly Balanced Bagging
# ------------------------
# While using a :class:`~imblearn.under_sampling.RandomUnderSampler` or
# :class:`~imblearn.over_sampling.RandomOverSampler` will create exactly the
# desired number of samples, it does not follow the statistical spirit wanted
# in the bagging framework. The authors in [3]_ proposes to use a negative
# binomial distribution to compute the number of samples of the majority
# class to be selected and then perform a random under-sampling.
#
# Here, we illustrate this method by implementing a function in charge of
# resampling and use the :class:`~imblearn.FunctionSampler` to integrate it
# within a :class:`~imblearn.pipeline.Pipeline` and
# :class:`~sklearn.model_selection.cross_validate`.

# %%
from collections import Counter

import numpy as np

from imblearn import FunctionSampler


def roughly_balanced_bagging(X, y, replace=False):
    """Implementation of Roughly Balanced Bagging for binary problem."""
    # find the minority and majority classes
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    # compute the number of sample to draw from the majority class using
    # a negative binomial distribution
    n_minority_class = class_counts[minority_class]
    n_majority_resampled = np.random.negative_binomial(n=n_minority_class, p=0.5)

    # draw randomly with or without replacement
    majority_indices = np.random.choice(
        np.flatnonzero(y == majority_class),
        size=n_majority_resampled,
        replace=replace,
    )
    minority_indices = np.random.choice(
        np.flatnonzero(y == minority_class),
        size=n_minority_class,
        replace=replace,
    )
    indices = np.hstack([majority_indices, minority_indices])

    return X[indices], y[indices]


# Roughly Balanced Bagging
rbb = BalancedBaggingClassifier(
    sampler=FunctionSampler(func=roughly_balanced_bagging, kw_args={"replace": True})
)
cv_results = cross_validate(rbb, X, y, scoring="balanced_accuracy")

print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")


# %% [markdown]
# .. topic:: References:
#
#    .. [1] R. Maclin, and D. Opitz. "An empirical evaluation of bagging and
#           boosting." AAAI/IAAI 1997 (1997): 546-551.
#
#    .. [2] S. Wang, and X. Yao. "Diversity analysis on imbalanced data sets by
#           using ensemble models." 2009 IEEE symposium on computational
#           intelligence and data mining. IEEE, 2009.
#
#    .. [3] S. Hido, H. Kashima, and Y. Takahashi. "Roughly balanced bagging
#          for imbalanced data." Statistical Analysis and Data Mining: The ASA
#          Data Science Journal 2.5‐6 (2009): 412-426.
