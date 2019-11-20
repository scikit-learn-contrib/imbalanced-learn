"""
==========================================================
Fitting model on imbalanced datasets and how to fight bias
==========================================================

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

classes_count = y.value_counts()
classes_count

###############################################################################
# This dataset is only slightly imbalanced. To better highlight the effect of
# learning from an imbalanced dataset, we will increase its ratio to 30:1

from imblearn.datasets import make_imbalance

ratio = 30
df_res, y_res = make_imbalance(
    df, y, sampling_strategy={
        classes_count.idxmin(): classes_count.max() // ratio
    }
)
y_res.value_counts()

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

###############################################################################
# Strategies to learn from an imbalanced dataset
###############################################################################

###############################################################################
# We will first define a helper function which will train a given model
# and compute both accuracy and balanced accuracy. The results will be stored
# in a dataframe

import pandas as pd


def evaluate_classifier(clf, df_scores, clf_name=None):
    from sklearn.pipeline import Pipeline
    if clf_name is None:
        if isinstance(clf, Pipeline):
            clf_name = clf[-1].__class__.__name__
        else:
            clf_name = clf.__class__.__name__
    acc = clf.fit(X_train, y_train).score(X_test, y_test)
    y_pred = clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    clf_score = pd.DataFrame(
        {clf_name: [acc, bal_acc]},
        index=['Accuracy', 'Balanced accuracy']
    )
    df_scores = pd.concat([df_scores, clf_score], axis=1).round(decimals=3)
    return df_scores


# Let's define an empty dataframe to store the results
df_scores = pd.DataFrame()

###############################################################################
# Dummy baseline
# ..............
#
# Before to train a real machine learning model, we can store the results
# obtained with our `DummyClassifier`.

df_scores = evaluate_classifier(dummy_clf, df_scores, "Dummy")
df_scores

###############################################################################
# Linear classifier baseline
# ..........................
#
# We will create a machine learning pipeline using a `LogisticRegression`
# classifier. In this regard, we will need to one-hot encode the categorical
# columns and standardized the numerical columns before to inject the data into
# the `LogisticRegression` classifier.
#
# First, we define our numerical and categorical pipelines.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

num_pipe = make_pipeline(
    StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True)
)
cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(handle_unknown="ignore")
)

###############################################################################
# Then, we can create a preprocessor which will dispatch the categorical
# columns to the categorical pipeline and the numerical columns to the
# numerical pipeline

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector

preprocessor_linear = ColumnTransformer(
    [("num-pipe", num_pipe, selector(dtype_include=np.number)),
     ("cat-pipe", cat_pipe, selector(dtype_include=pd.CategoricalDtype))],
    n_jobs=2
)

###############################################################################
# Finally, we connect our preprocessor with our `LogisticRegression`. We can
# then evaluate our model.

from sklearn.linear_model import LogisticRegression

lr_clf = make_pipeline(
    preprocessor_linear, LogisticRegression(max_iter=1000)
)
df_scores = evaluate_classifier(lr_clf, df_scores, "LR")
df_scores

###############################################################################
# We can see that our linear model is learning slightly better than our dummy
# baseline. However, it is impacted by the class imbalance.
#
# We can verify that something similar is happening with a tree-based model
# such as `RandomForestClassifier`. With this type of classifier, we will not
# need to scale the numerical data, and we will only need to ordinal encode the
# categorical data.

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OrdinalEncoder()
)

preprocessor_tree = ColumnTransformer(
    [("num-pipe", num_pipe, selector(dtype_include=np.number)),
     ("cat-pipe", cat_pipe, selector(dtype_include=pd.CategoricalDtype))],
    n_jobs=2
)

rf_clf = make_pipeline(
    preprocessor_tree, RandomForestClassifier(random_state=42, n_jobs=2)
)

df_scores = evaluate_classifier(rf_clf, df_scores, "RF")
df_scores

###############################################################################
# The `RandomForestClassifier` is as well affected by the class imbalanced,
# slightly less than the linear model. Now, we will present different approach
# to improve the performance of these 2 models.
#
# Use `class_weight`
# ..................
#
# Most of the models in `scikit-learn` have a parameter `class_weight`. This
# parameter will affect the computation of the loss in linear model or the
# criterion in the tree-based model to penalize differently a false
# classification from the minority and majority class. We can set
# `class_weight="balanced"` such that the weight applied is inversely
# proportional to the class frequency. We test this parametrization in both
# linear model and tree-based model.

lr_clf.set_params(logisticregression__class_weight="balanced")
df_scores = evaluate_classifier(
    lr_clf, df_scores, "LR with class weight"
)
df_scores

###############################################################################
#

rf_clf.set_params(randomforestclassifier__class_weight="balanced")
df_scores = evaluate_classifier(
    rf_clf, df_scores, "RF with class weight"
)
df_scores

###############################################################################
# We can see that using `class_weight` was really effective for the linear
# model, alleviating the issue of learning from imbalanced classes. However,
# the `RandomForestClassifier` is still biased toward the majority class,
# mainly due to the criterion which is not suited enough to fight the class
# imbalance.
#
# Resample the training set during learning
# .........................................
#
# Another way is to resample the training set by under-sampling or
# over-sampling some of the samples. `imbalanced-learn` provides some samplers
# to do such processing.

from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import RandomUnderSampler

lr_clf = make_pipeline_with_sampler(
    preprocessor_linear,
    RandomUnderSampler(random_state=42),
    LogisticRegression(max_iter=1000)
)
df_scores = evaluate_classifier(
    lr_clf, df_scores, "LR with under-sampling"
)
df_scores

###############################################################################
#

rf_clf = make_pipeline_with_sampler(
    preprocessor_tree,
    RandomUnderSampler(random_state=42),
    RandomForestClassifier(random_state=42, n_jobs=2)
)

df_scores = evaluate_classifier(
    rf_clf, df_scores, "RF with under-sampling"
)
df_scores

###############################################################################
# Applying a random under-sampler before the training of the linear model or
# random forest, allows to not focus on the majority class at the cost of
# making more mistake for samples in the majority class (i.e. decreased
# accuracy).
#
# We could apply any type of samplers and find which sampler is working best
# on the current dataset.
#
# Instead, we will present another way by using classifiers which will apply
# sampling internally.
#
# Use of `BalancedRandomForestClassifier` and `BalancedBaggingClassifier`
# .......................................................................
#
# We already showed that random under-sampling can be effective on decision
# tree. However, instead of under-sampling once the dataset, one could
# under-sample the original dataset before to take a bootstrap sample. This is
# the base of the `BalancedRandomForestClassifier` and
# `BalancedBaggingClassifier`.

from imblearn.ensemble import BalancedRandomForestClassifier

rf_clf = make_pipeline(
    preprocessor_tree,
    BalancedRandomForestClassifier(random_state=42, n_jobs=2)
)

df_scores = evaluate_classifier(rf_clf, df_scores, "Balanced RF")
df_scores

###############################################################################
# The performance with the `BalancedRandomForestClassifier` is better than
# applying a single random under-sampling. We will use a gradient-boosting
# classifier within a `BalancedBaggingClassifier`.

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.ensemble import BalancedBaggingClassifier

bag_clf = make_pipeline(
    preprocessor_tree,
    BalancedBaggingClassifier(
        base_estimator=HistGradientBoostingClassifier(random_state=42),
        n_estimators=10, random_state=42, n_jobs=2
    )
)

df_scores = evaluate_classifier(
    bag_clf, df_scores, "Balanced bagging"
)
df_scores

###############################################################################
# This last approach is the most effective. The different under-sampling allows
# to bring some diversity for the different GBDT to learn and not focus on a
# portion of the majority class.
#
# We will repeat the same experiment but with a ratio of 100:1 and make a
# similar analysis.

###############################################################################
# Increase imbalanced ratio
###############################################################################

ratio = 100
df_res, y_res = make_imbalance(
    df, y, sampling_strategy={
        classes_count.idxmin(): classes_count.max() // ratio
    }
)
X_train, X_test, y_train, y_test = train_test_split(
    df_res, y_res, stratify=y_res, random_state=42
)

df_scores = pd.DataFrame()
df_scores = evaluate_classifier(dummy_clf, df_scores, "Dummy")
lr_clf = make_pipeline(
    preprocessor_linear, LogisticRegression(max_iter=1000)
)
df_scores = evaluate_classifier(lr_clf, df_scores, "LR")
rf_clf = make_pipeline(
    preprocessor_tree, RandomForestClassifier(random_state=42, n_jobs=2)
)
df_scores = evaluate_classifier(rf_clf, df_scores, "RF")
lr_clf.set_params(logisticregression__class_weight="balanced")
df_scores = evaluate_classifier(
    lr_clf, df_scores, "LR with class weight"
)
rf_clf.set_params(randomforestclassifier__class_weight="balanced")
df_scores = evaluate_classifier(
    rf_clf, df_scores, "RF with class weight"
)
lr_clf = make_pipeline_with_sampler(
    preprocessor_linear,
    RandomUnderSampler(random_state=42),
    LogisticRegression(max_iter=1000)
)
df_scores = evaluate_classifier(
    lr_clf, df_scores, "LR with under-sampling"
)
rf_clf = make_pipeline_with_sampler(
    preprocessor_tree,
    RandomUnderSampler(random_state=42),
    RandomForestClassifier(random_state=42, n_jobs=2)
)
df_scores = evaluate_classifier(
    rf_clf, df_scores, "RF with under-sampling"
)
rf_clf = make_pipeline(
    preprocessor_tree,
    BalancedRandomForestClassifier(random_state=42, n_jobs=2)
)
df_scores = evaluate_classifier(rf_clf, df_scores)
df_scores = evaluate_classifier(
    bag_clf, df_scores, "Balanced bagging"
)
df_scores

###############################################################################
# When we analyse the results, we can draw similar conclusions than in the
# previous discussion. However, we can observe that the strategy
# `class_weight="balanced"` does not improve the performance when using a
# `RandomForestClassifier`. A resampling is indeed required. The most effective
# method remains the `BalancedBaggingClassifier` using a GBDT as a base
# learner.
