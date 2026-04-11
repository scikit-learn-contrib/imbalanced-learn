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

ratio, pos_label = 30, ">50K"
df_res, y_res = make_imbalance(
    df,
    y,
    sampling_strategy={classes_count.idxmin(): classes_count.max() // ratio},
)
y_res.value_counts()

# %% [markdown]
# We will use `skore.evaluate` to get an estimate of the test scores using
# cross-validation.
#
# As a baseline, we could use a classifier which will always predict the
# majority class independently of the features provided.

# %%
import skore
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
report = skore.evaluate(dummy_clf, df_res, y_res, splitter=5, pos_label=pos_label)
report.metrics.summarize().frame()

# %% [markdown]
# Strategies to learn from an imbalanced dataset
# -----------------------------------------------
#
# We will compare different strategies to learn from an imbalanced dataset by
# collecting all estimators and evaluating each using `skore.evaluate` to
# compute cross-validated metrics.
#
# Dummy baseline
# ..............
#
# Before to train a real machine learning model, we can store the results
# obtained with our :class:`~sklearn.dummy.DummyClassifier`.

# %%
estimators = [("Dummy classifier", dummy_clf)]

# %% [markdown]
# Linear classifier baseline
# ..........................
#
# We use `skrub.tabular_pipeline` to create a machine learning pipeline with
# proper preprocessing automatically adapted to the estimator. For a
# :class:`~sklearn.linear_model.LogisticRegression`, it will automatically
# handle missing values, encode categorical columns, and scale numerical
# columns.

# %%
from sklearn.linear_model import LogisticRegression
from skrub import tabular_pipeline

lr_clf = tabular_pipeline(LogisticRegression(max_iter=1000))
estimators.append(("Logistic regression", lr_clf))

# %% [markdown]
# We can verify that something similar is happening with a tree-based model
# such as :class:`~sklearn.ensemble.RandomForestClassifier`. `tabular_pipeline`
# will automatically adapt the preprocessing for tree-based models (e.g. no
# scaling needed).

# %%
from sklearn.ensemble import RandomForestClassifier

rf_clf = tabular_pipeline(RandomForestClassifier(random_state=42, n_jobs=2))
estimators.append(("Random forest", rf_clf))

# %% [markdown]
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
lr_clf_balanced = tabular_pipeline(
    LogisticRegression(max_iter=1000, class_weight="balanced")
)
estimators.append(("Logistic regression with balanced class weights", lr_clf_balanced))

rf_clf_balanced = tabular_pipeline(
    RandomForestClassifier(random_state=42, n_jobs=2, class_weight="balanced")
)
estimators.append(("Random forest with balanced class weights", rf_clf_balanced))

# %% [markdown]
# Resample the training set during learning
# .........................................
#
# Another way is to resample the training set by under-sampling or
# over-sampling some of the samples. `imbalanced-learn` provides some samplers
# to do such processing.
#
# We need to use the `imbalanced-learn` pipeline to properly handle the
# samplers within the pipeline. We insert the sampler before the final
# estimator in the pipeline created by `skrub.tabular_pipeline`.

# %%
from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import RandomUnderSampler

# We extract the preprocessing steps and the estimator from the tabular
# pipeline and insert the sampler before the estimator.
lr_clf_undersampled = make_pipeline_with_sampler(
    *lr_clf[:-1], RandomUnderSampler(random_state=42), lr_clf[-1]
)
estimators.append(("Under-sampling + Logistic regression", lr_clf_undersampled))

rf_clf_undersampled = make_pipeline_with_sampler(
    *rf_clf[:-1], RandomUnderSampler(random_state=42), rf_clf[-1]
)
estimators.append(("Under-sampling + Random forest", rf_clf_undersampled))

# %% [markdown]
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

brf_clf = tabular_pipeline(
    BalancedRandomForestClassifier(
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        random_state=42,
        n_jobs=2,
    )
)
estimators.append(("Balanced random forest", brf_clf))

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.ensemble import BalancedBaggingClassifier

bag_clf = tabular_pipeline(
    BalancedBaggingClassifier(
        estimator=HistGradientBoostingClassifier(random_state=42),
        n_estimators=10,
        random_state=42,
        n_jobs=2,
    )
)
estimators.append(("Balanced bag of histogram gradient boosting", bag_clf))

# %% [markdown]
# Now, we can use `skore.evaluate` to evaluate each estimator with
# cross-validation and collect all results in a single dataframe.

# %%
report = skore.evaluate(
    [est for _, est in estimators], df_res, y_res, splitter=5, pos_label=pos_label
)
report

# %%
results = report.metrics.summarize().frame(favorability=False)
results.rename(
    {
        previous_name: new_name
        for previous_name, (new_name, _) in zip(
            results.columns.get_level_values(1), estimators
        )
    },
    axis="columns",
    level=1,
)

# %% [markdown]
# This last approach is the most effective. The different under-sampling allows
# to bring some diversity for the different GBDT to learn and not focus on a
# portion of the majority class.
