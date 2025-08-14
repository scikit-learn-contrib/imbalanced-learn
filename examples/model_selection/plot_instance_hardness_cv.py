"""
====================================================
Distribute hard-to-classify datapoints over CV folds
====================================================

'Instance hardness' refers to the difficulty to classify an instance. The way
hard-to-classify instances are distributed over train and test sets has
significant effect on the test set performance metrics. In this example we
show how to deal with this problem. We are making the comparison with normal
:class:`~sklearn.model_selection.StratifiedKFold` cross-validation splitter.
"""

# Authors: Frits Hermans, https://fritshermans.github.io
# License: MIT

# %%
print(__doc__)

# %%
# Create an imbalanced dataset with instance hardness
# ---------------------------------------------------
#
# We create an imbalanced dataset with using scikit-learn's
# :func:`~sklearn.datasets.make_blobs` function and set the class imbalance ratio to
# 5%.
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=[950, 50], centers=((-3, 0), (3, 0)), random_state=10)
_ = plt.scatter(X[:, 0], X[:, 1], c=y)

# %%
# To introduce instance hardness in our dataset, we add some hard to classify samples:
X_hard, y_hard = make_blobs(
    n_samples=10, centers=((3, 0), (-3, 0)), cluster_std=1, random_state=10
)
X, y = np.vstack((X, X_hard)), np.hstack((y, y_hard))
_ = plt.scatter(X[:, 0], X[:, 1], c=y)

# %%
# Compare cross validation scores using `StratifiedKFold` and `InstanceHardnessCV`
# --------------------------------------------------------------------------------
#
# Now, we want to assess a linear predictive model. Therefore, we should use
# cross-validation. The most important concept with cross-validation is to create
# training and test splits that are representative of the the data in production to have
# statistical results that one can expect in production.
#
# By applying a standard :class:`~sklearn.model_selection.StratifiedKFold`
# cross-validation splitter, we do not control in which fold the hard-to-classify
# samples will be.
#
# The :class:`~imblearn.model_selection.InstanceHardnessCV` splitter allows to
# control the distribution of the hard-to-classify samples over the folds.
#
# Let's make an experiment to compare the results that we get with both splitters.
# We use a :class:`~sklearn.linear_model.LogisticRegression` classifier and
# :func:`~sklearn.model_selection.cross_validate` to calculate the cross validation
# scores. We use average precision for scoring.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from imblearn.model_selection import InstanceHardnessCV

logistic_regression = LogisticRegression()

results = {}
for cv in (
    StratifiedKFold(n_splits=5, shuffle=True, random_state=10),
    InstanceHardnessCV(estimator=LogisticRegression()),
):
    result = cross_validate(
        logistic_regression,
        X,
        y,
        cv=cv,
        scoring="average_precision",
    )
    results[cv.__class__.__name__] = result["test_score"]
results = pd.DataFrame(results)

# %%
ax = results.plot.box(vert=False, whis=[0, 100])
_ = ax.set(
    xlabel="Average precision",
    title="Cross validation scores with different splitters",
    xlim=(0, 1),
)

# %%
# The boxplot shows that the :class:`~imblearn.model_selection.InstanceHardnessCV`
# splitter results in less variation of average precision than
# :class:`~sklearn.model_selection.StratifiedKFold` splitter. When doing
# hyperparameter tuning or feature selection using a wrapper method (like
# :class:`~sklearn.feature_selection.RFECV`) this will give more stable results.
