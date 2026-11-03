"""
=============================================
Compare ranking and calibration with sampling
=============================================

Resampling is not automatically an improvement in discrimination or probability
calibration. Compare a baseline classifier against class weighting and SMOTE on
an untouched test set. The example is illustrative, not a universal ranking of
methods. Sampling is performed only within the training pipeline to prevent
information leaking from the test set.
"""

# Authors: Community contributors
# License: MIT

# %%
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

# %% [markdown]
# Create an imbalanced binary classification task and leave the original
# test distribution unchanged for all model evaluations.

# %%
X, y = make_classification(
    n_samples=5000,
    n_features=12,
    n_informative=4,
    n_redundant=2,
    weights=[0.94, 0.06],
    class_sep=0.9,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

models = {
    "Unweighted": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Class-weighted": make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    ),
    "SMOTE": make_imb_pipeline(
        StandardScaler(), SMOTE(random_state=42), LogisticRegression(max_iter=1000)
    ),
}

# %% [markdown]
# ROC-AUC measures ranking discrimination. Brier score evaluates probabilistic
# predictions (lower is better), but is not a calibration-only metric. The
# reliability curves compare predicted probabilities with observed frequencies.

# %%
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print(
        f"{name}: ROC-AUC={roc_auc_score(y_test, proba):.3f}, "
        f"Brier={brier_score_loss(y_test, proba):.3f}"
    )
    CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, name=name, ax=ax)

ax.set_title("Calibration on the unchanged test distribution")
fig.tight_layout()
plt.show()

# %% [markdown]
# Interpreting the comparison
# --------------------------
# An improvement in ROC-AUC does not guarantee better probability estimates.
# Brier score reflects more than calibration alone, so inspect it together with
# the reliability diagram. These observations describe this synthetic dataset
# and should not be generalized to every classifier or sampling strategy.
