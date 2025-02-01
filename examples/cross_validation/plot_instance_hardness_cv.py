"""
===================================================
Distribute hard-to-classify datapoint over CV folds
===================================================

'Instance hardness' refers to the difficulty to classify an instance. The way
hard-to-classify instances are distributed over train and test sets has
significant effect on the test set performance metrics. In this example we
show how to deal with this problem. We are making the comparison with normal
StratifiedKFold cv splitting.
"""

# Authors: Frits Hermans, https://fritshermans.github.io
# License: MIT

# %%
print(__doc__)

# %% [markdown]
# Create an imbalanced dataset with instance hardness
# ---------------------------------------------------
#
# We will create an imbalanced dataset with using scikit-learn's `make_blobs`
# function and the `make_imbalance` function. The imbalancedness is set to
# 0.1; only 10% of the labels is positive.


import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

from imblearn.datasets import make_imbalance

X, y = make_blobs(n_samples=1000, centers=((-3, 0), (3, 0)), random_state=10)


# %%
def sampling_strategy(ratio):
    def strategy(y):
        return {0: sum(y), 1: int(ratio * sum(y) / (1 - ratio))}

    return strategy


X, y = make_imbalance(X, y, sampling_strategy=sampling_strategy(0.1), random_state=10)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# %%
# To introduce instance hardness in our dataset, we flip the labels at the
# boundaries of the feature space
y[np.argsort(X[:, 0])[:5]] = 1
y[np.argsort(X[:, 0])[-5:]] = 0
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# %% [markdown]
# Compare cross validation scores using StratifiedKFold and InstanceHardnessCV
# ----------------------------------------------------------------------------
#
# We calculate cross validation scores using `cross_validate` and a
# `LogisticRegression` classifier. We compare the results using a
# `StratifiedKFold` cv splitter and an `InstanceHardnessCV` splitter.
# As we are dealing with an imbalanced classification problem, we
# use `average_precision` for scoring.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from imblearn.cross_validation import InstanceHardnessCV

# %%
clf = LogisticRegression()

# %%
skf_cv = StratifiedKFold(n_splits=5)
skf_result = cross_validate(clf, X, y, cv=skf_cv, scoring="average_precision")

# %%
ih_cv = InstanceHardnessCV(n_splits=5, random_state=10)
ih_result = cross_validate(clf, X, y, cv=ih_cv)

# %%
# The boxplot below shows that the `InstanceHardnessCV` splitter results
# in less variation of average precision than `StratifiedKFold` splitter.
# When doing hyperparameter tuning or feature selection using a wrapper
# method (like `RFECV`) this will give more stable results.

import pandas as pd

ax = (
    pd.concat(
        (pd.DataFrame(skf_result), pd.DataFrame(ih_result)),
        axis=1,
        keys=["StratifiedKFold", "InstanceHardnessCV"],
    )
    .swaplevel(axis="columns")["test_score"]
    .plot.box(
        color={"whiskers": "black", "medians": "black", "caps": "black"}, vert=False
    )
)
plt.xlabel("Average precision")
_ = plt.title("Test score via cross-validation")
plt.tight_layout()
plt.show()
