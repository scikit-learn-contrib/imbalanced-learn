import numpy as np
import pytest

datasets = pytest.importorskip("datasets")
from datasets import load_dataset  
from imblearn.pipeline import Pipeline  
from imblearn.under_sampling import RandomUnderSampler  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.svm import LinearSVC  


def _small_split(n_train=900, n_test=300):
    ds = load_dataset("tweet_eval", "sentiment")
    X = [x["text"] for x in ds["train"]][: n_train + n_test]
    y = np.array([x["label"] for x in ds["train"]][: n_train + n_test], dtype=int)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def test_pipeline_trains_and_predicts():
    Xtr, ytr, Xte, yte = _small_split()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000)),
        ("balance", RandomUnderSampler(random_state=0)),
        ("clf", LinearSVC()),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    assert len(pred) == len(yte)
    # predictions should be 0/1/2 labels
    assert set(np.unique(pred)).issubset({0, 1, 2})
