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

# %%
from sklearn.model_selection import cross_validate
from sklearn.ensemble import BaggingClassifier

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
# .. topic:: References:
#
#    .. [1] R. Maclin, and D. Opitz. "An empirical evaluation of bagging and
#           boosting." AAAI/IAAI 1997 (1997): 546-551.
#
#    .. [2] S. Wang, and X. Yao. "Diversity analysis on imbalanced data sets by
#           using ensemble models." 2009 IEEE symposium on computational
#           intelligence and data mining. IEEE, 2009.
