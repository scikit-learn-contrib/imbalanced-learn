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

# %%
print(__doc__)

# %% [markdown]
# Problem definition
# ------------------
#
# We are dropping the following features:
#
# - "fnlwgt": this feature was created while studying the "adult" dataset.
#   Thus, we will not use this feature which is not acquired during the survey.
# - "education-num": it is encoding the same information than "education".
#   Thus, we are removing one of these 2 features.

# %%
from sklearn.datasets import fetch_openml

df, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
df = df.drop(columns=["fnlwgt", "education-num"])

# %% [markdown]
# The "adult" dataset as a class ratio of about 3:1

# %%
classes_count = y.value_counts()
classes_count

# %% [markdown]
# This dataset is only slightly imbalanced. To better highlight the effect of
# learning from an imbalanced dataset, we will increase its ratio to 30:1

# %%
from imblearn.datasets import make_imbalance

ratio = 30
df_res, y_res = make_imbalance(
    df,
    y,
    sampling_strategy={classes_count.idxmin(): classes_count.max() // ratio},
)
y_res.value_counts()

# %% [markdown]
# We will perform a cross-validation evaluation to get an estimate of the test
# score.
#
# As a baseline, we could use a classifier which will always predict the
# majority class independently of the features provided.

from sklearn.dummy import DummyClassifier

# %%
from sklearn.model_selection import cross_validate

dummy_clf = DummyClassifier(strategy="most_frequent")
scoring = ["accuracy", "balanced_accuracy"]
cv_result = cross_validate(dummy_clf, df_res, y_res, scoring=scoring)
print(f"Accuracy score of a dummy classifier: {cv_result['test_accuracy'].mean():.3f}")

# %% [markdown]
# Instead of using the accuracy, we can use the balanced accuracy which will
# take into account the balancing issue.

# %%
print(
    f"Balanced accuracy score of a dummy classifier: "
    f"{cv_result['test_balanced_accuracy'].mean():.3f}"
)

# %% [markdown]
# Strategies to learn from an imbalanced dataset
# ----------------------------------------------
# We will use a dictionary and a list to continuously store the results of
# our experiments and show them as a pandas dataframe.

# %%
index = []
scores = {"Accuracy": [], "Balanced accuracy": []}

# %% [markdown]
# Dummy baseline
# ..............
#
# Before to train a real machine learning model, we can store the results
# obtained with our :class:`~sklearn.dummy.DummyClassifier`.

# %%
import pandas as pd

index += ["Dummy classifier"]
cv_result = cross_validate(dummy_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% [markdown]
# Linear classifier baseline
# ..........................
#
# We will create a machine learning pipeline using a
# :class:`~sklearn.linear_model.LogisticRegression` classifier. In this regard,
# we will need to one-hot encode the categorical columns and standardized the
# numerical columns before to inject the data into the
# :class:`~sklearn.linear_model.LogisticRegression` classifier.
#
# First, we define our numerical and categorical pipelines.

# %%
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_pipe = make_pipeline(
    StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True)
)
cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(handle_unknown="ignore"),
)

# %% [markdown]
# Then, we can create a preprocessor which will dispatch the categorical
# columns to the categorical pipeline and the numerical columns to the
# numerical pipeline

# %%
from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer

preprocessor_linear = make_column_transformer(
    (num_pipe, selector(dtype_include="number")),
    (cat_pipe, selector(dtype_include="category")),
    n_jobs=2,
)

# %% [markdown]
# Finally, we connect our preprocessor with our
# :class:`~sklearn.linear_model.LogisticRegression`. We can then evaluate our
# model.

# %%
from sklearn.linear_model import LogisticRegression

lr_clf = make_pipeline(preprocessor_linear, LogisticRegression(max_iter=1000))

# %%
index += ["Logistic regression"]
cv_result = cross_validate(lr_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% [markdown]
# We can see that our linear model is learning slightly better than our dummy
# baseline. However, it is impacted by the class imbalance.
#
# We can verify that something similar is happening with a tree-based model
# such as :class:`~sklearn.ensemble.RandomForestClassifier`. With this type of
# classifier, we will not need to scale the numerical data, and we will only
# need to ordinal encode the categorical data.

from sklearn.ensemble import RandomForestClassifier

# %%
from sklearn.preprocessing import OrdinalEncoder

num_pipe = SimpleImputer(strategy="mean", add_indicator=True)
cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
)

preprocessor_tree = make_column_transformer(
    (num_pipe, selector(dtype_include="number")),
    (cat_pipe, selector(dtype_include="category")),
    n_jobs=2,
)

rf_clf = make_pipeline(
    preprocessor_tree, RandomForestClassifier(random_state=42, n_jobs=2)
)

# %%
index += ["Random forest"]
cv_result = cross_validate(rf_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% [markdown]
# The :class:`~sklearn.ensemble.RandomForestClassifier` is as well affected by
# the class imbalanced, slightly less than the linear model. Now, we will
# present different approach to improve the performance of these 2 models.
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

# %%
lr_clf.set_params(logisticregression__class_weight="balanced")

index += ["Logistic regression with balanced class weights"]
cv_result = cross_validate(lr_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %%
rf_clf.set_params(randomforestclassifier__class_weight="balanced")

index += ["Random forest with balanced class weights"]
cv_result = cross_validate(rf_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% [markdown]
# We can see that using `class_weight` was really effective for the linear
# model, alleviating the issue of learning from imbalanced classes. However,
# the :class:`~sklearn.ensemble.RandomForestClassifier` is still biased toward
# the majority class, mainly due to the criterion which is not suited enough to
# fight the class imbalance.
#
# Resample the training set during learning
# .........................................
#
# Another way is to resample the training set by under-sampling or
# over-sampling some of the samples. `imbalanced-learn` provides some samplers
# to do such processing.

# %%
from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import RandomUnderSampler

lr_clf = make_pipeline_with_sampler(
    preprocessor_linear,
    RandomUnderSampler(random_state=42),
    LogisticRegression(max_iter=1000),
)

# %%
index += ["Under-sampling + Logistic regression"]
cv_result = cross_validate(lr_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %%
rf_clf = make_pipeline_with_sampler(
    preprocessor_tree,
    RandomUnderSampler(random_state=42),
    RandomForestClassifier(random_state=42, n_jobs=2),
)

# %%
index += ["Under-sampling + Random forest"]
cv_result = cross_validate(rf_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% [markdown]
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
# Use of specific balanced algorithms from imbalanced-learn
# .........................................................
#
# We already showed that random under-sampling can be effective on decision
# tree. However, instead of under-sampling once the dataset, one could
# under-sample the original dataset before to take a bootstrap sample. This is
# the base of the :class:`imblearn.ensemble.BalancedRandomForestClassifier` and
# :class:`~imblearn.ensemble.BalancedBaggingClassifier`.

# %%
from imblearn.ensemble import BalancedRandomForestClassifier

rf_clf = make_pipeline(
    preprocessor_tree,
    BalancedRandomForestClassifier(
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        random_state=42,
        n_jobs=2,
    ),
)

# %%
index += ["Balanced random forest"]
cv_result = cross_validate(rf_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% [markdown]
# The performance with the
# :class:`~imblearn.ensemble.BalancedRandomForestClassifier` is better than
# applying a single random under-sampling. We will use a gradient-boosting
# classifier within a :class:`~imblearn.ensemble.BalancedBaggingClassifier`.

from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.ensemble import BalancedBaggingClassifier

bag_clf = make_pipeline(
    preprocessor_tree,
    BalancedBaggingClassifier(
        estimator=HistGradientBoostingClassifier(random_state=42),
        n_estimators=10,
        random_state=42,
        n_jobs=2,
    ),
)

index += ["Balanced bag of histogram gradient boosting"]
cv_result = cross_validate(bag_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
df_scores

# %% [markdown]
# This last approach is the most effective. The different under-sampling allows
# to bring some diversity for the different GBDT to learn and not focus on a
# portion of the majority class.
