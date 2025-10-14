import os
import sys
import numpy as np
import pytest

datasets = pytest.importorskip("datasets")

# Import the example as a module (pytest adds repo root to sys.path)
from examples.text_sentiment_svm_with_resampling import (
    main,
    load_tweet_eval,
)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loader_reproducible_small():
    """Same seed -> identical splits (reproducibility)."""
    X1, y1, Xt1, Yt1 = load_tweet_eval(max_samples=800, random_state=42)
    X2, y2, Xt2, Yt2 = load_tweet_eval(max_samples=800, random_state=42)
    assert X1 == X2
    assert np.array_equal(y1, y2)
    assert Xt1 == Xt2
    assert np.array_equal(Yt1, Yt2)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_smoke_predicts_labels_small():
    """End-to-end: pipeline trains and predicts on a tiny slice."""
    Xtr, ytr, Xte, yte = load_tweet_eval(max_samples=800, random_state=0)
    # Build the same pipeline as in the example
    from imblearn.pipeline import Pipeline
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
        ("balance", RandomUnderSampler(random_state=0)),
        ("clf", LinearSVC()),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    assert len(pred) == len(yte)
    # Predictions must be in the expected label set {0,1,2}
    assert set(np.unique(pred)).issubset({0, 1, 2})

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cli_saves_plot(tmp_path):
    """CLI: --plot should create the confusion matrix image."""
    out = tmp_path / "cm.png"
    main(["--plot", "--max-samples", "800", "--output", str(out)])
    assert out.exists() and out.stat().st_size > 0

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cli_no_plot_no_file(tmp_path):
    """CLI: without --plot, no image should be created."""
    out = tmp_path / "cm.png"
    if out.exists():
        os.remove(out)
    main(["--max-samples", "500", "--output", str(out)])
    assert not out.exists()
