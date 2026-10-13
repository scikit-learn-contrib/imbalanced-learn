"""Tests for the TF-IDF + sampler + linear SVM pattern used in the text example."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline

# Imbalanced toy corpus: 20 negative, 6 neutral, 4 positive documents, each
# class using its own vocabulary so that the problem is linearly separable.
X = (
    [f"terrible awful delayed flight {i}" for i in range(20)]
    + [f"flight arrived tuesday gate {i}" for i in range(6)]
    + [f"great amazing friendly crew {i}" for i in range(4)]
)
y = np.array(["negative"] * 20 + ["neutral"] * 6 + ["positive"] * 4)
LABELS = {"negative", "neutral", "positive"}


def _make_models():
    return {
        "baseline": make_pipeline(TfidfVectorizer(), LinearSVC()),
        "class_weight": make_pipeline(
            TfidfVectorizer(), LinearSVC(class_weight="balanced")
        ),
        "oversampling": make_pipeline(
            TfidfVectorizer(), RandomOverSampler(random_state=0), LinearSVC()
        ),
    }


@pytest.mark.parametrize("name", ["baseline", "class_weight", "oversampling"])
def test_pipeline_predicts_valid_labels(name):
    model = _make_models()[name].fit(X, y)
    y_pred = model.predict(["awful delayed flight", "great friendly crew"])
    assert set(y_pred) <= LABELS


@pytest.mark.parametrize("name", ["baseline", "class_weight", "oversampling"])
def test_separable_toy_data(name):
    model = _make_models()[name].fit(X, y)
    assert model.predict(["terrible awful delayed"])[0] == "negative"
    assert model.predict(["great amazing friendly"])[0] == "positive"


def test_oversampler_balances_classes():
    X_vec = TfidfVectorizer().fit_transform(X)
    _, y_res = RandomOverSampler(random_state=0).fit_resample(X_vec, y)
    _, counts = np.unique(y_res, return_counts=True)
    assert len(set(counts)) == 1
    assert counts[0] == 20


def test_predict_does_not_resample():
    model = _make_models()["oversampling"].fit(X, y)
    X_new = ["awful flight", "great crew", "flight arrived"]
    assert len(model.predict(X_new)) == len(X_new)
