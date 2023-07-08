"""
=======================================
Metrics specific to imbalanced learning
=======================================

Specific metrics have been developed to evaluate classifier which
has been trained using imbalanced data. :mod:`imblearn` provides mainly
two additional metrics which are not implemented in :mod:`sklearn`: (i)
geometric mean and (ii) index balanced accuracy.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

# %%
print(__doc__)

RANDOM_STATE = 42

# %% [markdown]
# First, we will generate some imbalanced dataset.

# %%
from sklearn.datasets import make_classification

X, y = make_classification(
    n_classes=3,
    class_sep=2,
    weights=[0.1, 0.9],
    n_informative=10,
    n_redundant=1,
    flip_y=0,
    n_features=20,
    n_clusters_per_class=4,
    n_samples=5000,
    random_state=RANDOM_STATE,
)

# %% [markdown]
# We will split the data into a training and testing set.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=RANDOM_STATE
)

# %% [markdown]
# We will create a pipeline made of a :class:`~imblearn.over_sampling.SMOTE`
# over-sampler followed by a :class:`~sklearn.linear_model.LogisticRegression`
# classifier.

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

# %%
from imblearn.pipeline import make_pipeline

model = make_pipeline(
    StandardScaler(),
    SMOTE(random_state=RANDOM_STATE),
    LogisticRegression(max_iter=10_000, random_state=RANDOM_STATE),
)

# %% [markdown]
# Now, we will train the model on the training set and get the prediction
# associated with the testing set. Be aware that the resampling will happen
# only when calling `fit`: the number of samples in `y_pred` is the same than
# in `y_test`.

# %%
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %% [markdown]
# The geometric mean corresponds to the square root of the product of the
# sensitivity and specificity. Combining the two metrics should account for
# the balancing of the dataset.

# %%
from imblearn.metrics import geometric_mean_score

print(f"The geometric mean is {geometric_mean_score(y_test, y_pred):.3f}")

# %% [markdown]
# The index balanced accuracy can transform any metric to be used in
# imbalanced learning problems.

# %%
from imblearn.metrics import make_index_balanced_accuracy

alpha = 0.1
geo_mean = make_index_balanced_accuracy(alpha=alpha, squared=True)(geometric_mean_score)

print(
    f"The IBA using alpha={alpha} and the geometric mean: "
    f"{geo_mean(y_test, y_pred):.3f}"
)

# %%
alpha = 0.5
geo_mean = make_index_balanced_accuracy(alpha=alpha, squared=True)(geometric_mean_score)

print(
    f"The IBA using alpha={alpha} and the geometric mean: "
    f"{geo_mean(y_test, y_pred):.3f}"
)
