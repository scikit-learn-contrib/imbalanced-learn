"""
=======================================
Metrics specific to imbalanced learning
=======================================

Specific metrics have been developed to evaluate classifier which
has been trained using imbalanced data. `imblearn` provides mainly
two additional metrics which are not implemented in `sklearn`: (i)
geometric mean and (ii) index balanced accuracy.
"""

from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from imblearn import over_sampling as os
from imblearn import pipeline as pl
from imblearn.metrics import (geometric_mean_score,
                              make_index_balanced_accuracy)

print(__doc__)

RANDOM_STATE = 42

# Generate a dataset
X, y = datasets.make_classification(n_classes=3, class_sep=2,
                                    weights=[0.1, 0.9], n_informative=10,
                                    n_redundant=1, flip_y=0, n_features=20,
                                    n_clusters_per_class=4, n_samples=5000,
                                    random_state=RANDOM_STATE)

pipeline = pl.make_pipeline(os.SMOTE(random_state=RANDOM_STATE),
                            LinearSVC(random_state=RANDOM_STATE))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=RANDOM_STATE)

# Train the classifier with balancing
pipeline.fit(X_train, y_train)

# Test the classifier and get the prediction
y_pred_bal = pipeline.predict(X_test)

###############################################################################
# The geometric mean corresponds to the square root of the product of the
# sensitivity and specificity. Combining the two metrics should account for
# the balancing of the dataset.

print('The geometric mean is {}'.format(geometric_mean_score(
    y_test,
    y_pred_bal)))

###############################################################################
# The index balanced accuracy can transform any metric to be used in
# imbalanced learning problems.

alpha = 0.1
geo_mean = make_index_balanced_accuracy(alpha=alpha, squared=True)(
    geometric_mean_score)

print('The IBA using alpha = {} and the geometric mean: {}'.format(
    alpha, geo_mean(
        y_test,
        y_pred_bal)))

alpha = 0.5
geo_mean = make_index_balanced_accuracy(alpha=alpha, squared=True)(
    geometric_mean_score)

print('The IBA using alpha = {} and the geometric mean: {}'.format(
    alpha, geo_mean(
        y_test,
        y_pred_bal)))
