"""
========================================================================
Model fitting on imbalanced dataset and comparison of methods to improve
performance
========================================================================

This example illustrates the problem induced by learning on datasets having
imbalanced classes. Subsequently, we compare different approaches alleviating
these negative effects.

"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

print(__doc__)

###############################################################################
# Problem definition
###############################################################################

from sklearn.datasets import fetch_openml

df, y = fetch_openml('adult', version=2, as_frame=True, return_X_y=True)
# we are dropping the following features:
# - "fnlwgt": this feature was created while studying the "adult" dataset.
#   Thus, we will not use this feature which is not acquired during the survey.
# - "education-num": it is encoding the same information than "education".
#   Thus, we are removing one of these 2 features.
df = df.drop(columns=['fnlwgt', 'education-num'])

###############################################################################
# The "adult" dataset as a class ratio of about 3:1

from collections import Counter

classes_count = y.value_counts()
print(f"Classes information:\n{classes_count}")

###############################################################################
# This dataset is only slightly imbalanced. To better highlight the effect of
# learning from imbalanced dataset, we will increase this ratio to 30:1

from imblearn.datasets import make_imbalance

ratio = 30
df_res, y_res = make_imbalance(
    df, y, sampling_strategy={
        classes_count.idxmin(): classes_count.max() // ratio
    }
)

###############################################################################
# For the rest of the notebook, we will make a single split to get training
# and testing data. Note that you should use cross-validation to have an
# estimate of the performance variation in practice.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df_res, y_res, stratify=y_res, random_state=42
)

###############################################################################
# As a baseline, we could use a classifier which will always predict the
# majority class independently of the features provided.

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
score = dummy_clf.fit(X_train, y_train).score(X_test, y_test)
print(f"Accuracy score of a dummy classifier: {score:.3f}")

##############################################################################
# Instead of using the accuracy, we can use the balanced accuracy which will
# take into account the balancing issue.

from sklearn.metrics import balanced_accuracy_score

y_pred = dummy_clf.predict(X_test)
score = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced accuracy score of a dummy classifier: {score:.3f}")
