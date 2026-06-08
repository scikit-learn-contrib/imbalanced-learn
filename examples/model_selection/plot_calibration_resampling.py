"""
====================================================
Effect of Resampling on Probability Calibration for Classifiers
====================================================

With this example we illustrate how resampling a data set (like Under-sampling) can affect
the calibration of a classifier's predicted probabilities , and how can we fix this issue using
:class:`~sklearn.calibration.CalibratedClassifierCV`

When we resample a dataset so we can balance it , we change the prior probabilities
of the classes contained in the dataset.The model learns that some classes
are more frequent than they actually are and some classes are less frequent than they actually are.

This example shows:
1. The calibration curve of a model trained on the original dataset.
2. The resampled model's calibration curve, which is distorted
3. How to recover the probabilities using calibration
"""

# Authors: The imbalanced-learn developers
# License: MIT

# %%
# Create an imbalanced dataset with two classes using scikit-learn's
# :func:`sklearn.datasets.make_classification` function with 95-5 class ratio
# and split the data into training and testing sets (80-20) using
# :func:`sklearn.model_selection.train_test_split.` function and set stratify = y
# because we want to keep the 95-5 class ratio.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X , y = make_classification(
    n_samples=10000,
    n_features=20,
    n_classes=2,
    weights=[0.95,0.05],
    )

X_train , X_test , y_train , y_test = train_test_split(
    X,y , test_size=0.2 , random_state=42 , stratify=y
    )

# %%
# The Problem: Resampling distorts probabilities
# -----------------------------------------------
# At first , we train a :class:`~sklearn.linear_model.LogisticRegression` classifier on
# the original data. Then, we train a second :class:`~sklearn.linear_model.LogisticRegression` classifier
# on data that has been undersampled to a 50-50 ratio using a :class:`imblearn.under_sampling.RandomUnderSampler`.

from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler

# Train Logistic Regression Model (Vanilla Model)
lr_original = LogisticRegression(random_state=42)
lr_original.fit(X_train,y_train)

# Train Resampled Model (Under-sampling)
under_sampler = RandomUnderSampler(random_state=42)
X_undersampled , y_undersampled = under_sampler.fit_resample(X_train, y_train)

lr_undersampled = LogisticRegression(random_state=42)
lr_undersampled.fit(X_undersampled, y_undersampled)

# %%
# We plot the calibration curves to compare the two models using :class:`~sklearn.calibration.CalibrationDisplay`.
# The diagonal line represents a perfectly calibrated model.

from sklearn.calibration import CalibrationDisplay

fig, ax = plt.subplots(figsize=(8,6))

CalibrationDisplay.from_estimator(
    lr_original, X_test, y_test, n_bins=10, name="Original model", ax=ax
)

CalibrationDisplay.from_estimator(
    lr_undersampled, X_test, y_test, n_bins=10, name="Undersampled model", ax=ax
)

plt.title("Calibration: Original vs Resampled")
plt.show()

# %%
# **Observation:**
# The resampled model's curve is significantly below the diagonal. It is obvious that the
# model is over-confident: it predicts high probabilities for the
# positive class, but the actual fraction of positives is much lower.

# %%
# The Solution: Probability Calibration
# ------------------------------------
# We use :class:`~sklearn.calibration.CalibratedClassifierCV`to calibrate the model.It is important
# to note that the calibrator needs to be trained to data with the real class distribution. Therefore we split
# the training set into two parts:
#
# ``X_model_train``: used for training and resampled
# ``X_calib``: used to train the calibrator (original distribution)

from sklearn.calibration import CalibratedClassifierCV

# Split the training set
X_model_train, X_calib, y_model_train, y_calib = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# Resample and train the model
X_undersampled2, y_undersampled2 = under_sampler.fit_resample(X_model_train, y_model_train)
lr_resampled = LogisticRegression(random_state=42)
lr_resampled.fit(X_undersampled2, y_undersampled2)

# Calibrate using the untouched set (X_calib,y_calib)
# We use method='sigmoid', good for logisitc regression and for few positives samples in the calib set
# We use cv='prefit' because the base model is already trained.
calibrated_model = CalibratedClassifierCV(
    lr_resampled, method="sigmoid", cv="prefit"
)
calibrated_model.fit(X_calib, y_calib)

# %%
# Comparing the Results
# ---------------------
# We plot the calibration curve of the fixed model

fig, ax = plt.subplots(figsize=(8, 6))

# Plot the undersampled uncalibrated model
CalibrationDisplay.from_estimator(
    lr_resampled, X_test, y_test, n_bins=10, name="Uncalibrated (Undersampled)", ax=ax
)

# Plot the new calibrated model
CalibrationDisplay.from_estimator(
    calibrated_model, X_test, y_test, n_bins=10, name="Calibrated Model", ax=ax
)

plt.title("Effect of Calibration on Resampled Model")
plt.show()

# %%
# We can also check that the calibration did not affect the model's ranking quality by
# checking that ROC AUC (discrimination power) has not been affected.
# In addition we can use the Brier Score metric to see the improvement in the
# probabilty accuracy. We will use :func:`sklearn.metrics.roc_auc_score` function and
# :func:`sklearn.metrics.brier_score_loss` function

from sklearn.metrics import roc_auc_score, brier_score_loss

# probability estimation for the train set for class 1 (minority)
uncalibrated_prob = lr_resampled.predict_proba(X_test)[:,1]
prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

print(f"ROC AUC (Uncalibrated): {roc_auc_score(y_test, uncalibrated_prob):.4f}")
print(f"ROC AUC (Calibrated):   {roc_auc_score(y_test, prob_calibrated):.4f}")
print("-" * 30)
print("For the Brier Score the smaller is the better")
print(f"Brier Score (Uncalibrated): {brier_score_loss(y_test, uncalibrated_prob):.4f}")
print(f"Brier Score (Calibrated):   {brier_score_loss(y_test, prob_calibrated):.4f}")