"""
=========================================
Resampling a Polars dataset in a pipeline
=========================================

Samplers preserve Polars column names and dtypes, including when a scikit-learn
transformer produces a Polars DataFrame. This example standardizes an imbalanced
dataset, generates minority-class samples with SMOTE, and fits a classifier.

Polars is optional: install it with ``pip install polars`` to run this example.
"""

# License: MIT

# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

try:
    import polars as pl
except ImportError:
    pl = None

# %% [markdown]
# Split before resampling so that synthetic samples cannot leak information
# from the test set. The target remains a named Polars Series.

# %%
if pl is None:
    print("Install polars to run this example: pip install polars")
else:
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_redundant=0,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X = pl.DataFrame(X, schema=["a", "b", "c", "d"], orient="row")
    y = pl.Series("target", y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )

    model = make_pipeline(
        StandardScaler().set_output(transform="polars"),
        SMOTE(random_state=42),
        LogisticRegression(),
    )
    model.fit(X_train, y_train)
    print("Feature names:", model[-1].feature_names_in_)
    print("Balanced accuracy:", balanced_accuracy_score(y_test, model.predict(X_test)))

    X_res, y_res = model[:-1].fit_resample(X_train, y_train)
    print("Resampled feature schema:", X_res.schema)
    print("Resampled class counts:", y_res.value_counts().to_dicts())
