"""
====================================================
Distribute hard-to-classify datapoints over CV folds
====================================================

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
# function and set the imbalancedness to 5%; only 5% of the labels is positive.


import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=[950,50], centers=((-3, 0), (3, 0)), random_state=10)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# %%
# To introduce instance hardness in our dataset, we add some hard to classify samples:
X_hard, y_hard = make_blobs(n_samples=10, centers=((3, 0), (-3, 0)),
                            cluster_std=1,
                            random_state=10)
X = np.vstack((X, X_hard))
y = np.hstack((y, y_hard))
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
clf = LogisticRegression(random_state=10)

# %%
skf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
skf_result = cross_validate(clf, X, y, cv=skf_cv, scoring="average_precision")

# %%
ih_cv = InstanceHardnessCV(n_splits=5, estimator=clf, random_state=10)
ih_result = cross_validate(clf, X, y, cv=ih_cv, scoring="average_precision")

# %%
# The boxplot below shows that the `InstanceHardnessCV` splitter results
# in less variation of average precision than `StratifiedKFold` splitter.
# When doing hyperparameter tuning or feature selection using a wrapper
# method (like `RFECV`) this will give more stable results.

# %%
plt.boxplot([skf_result['test_score'], ih_result['test_score']],
            tick_labels=["StratifiedKFold", "InstanceHardnessCV"], vert=False)
plt.xlabel('Average precision')
plt.tight_layout()
plt.show()
