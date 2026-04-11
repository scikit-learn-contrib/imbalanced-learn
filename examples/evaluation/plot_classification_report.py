"""
=============================================
Evaluate classification by compiling a report
=============================================

Specific metrics have been developed to evaluate classifier which has been
trained using imbalanced data. We use `skore.evaluate` to get a structured
report of the classifier performance.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT


import skore
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn import over_sampling as os
from imblearn import pipeline as pl

print(__doc__)

RANDOM_STATE = 42

# Generate a dataset
X, y = datasets.make_classification(
    n_classes=2,
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

pipeline = pl.make_pipeline(
    StandardScaler(),
    os.SMOTE(random_state=RANDOM_STATE),
    LogisticRegression(max_iter=10_000),
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

# Train the classifier with balancing
pipeline.fit(X_train, y_train)

# Evaluate the classifier using skore
report = skore.evaluate(pipeline, X_test, y_test, splitter="prefit")
report.metrics.summarize().frame()
