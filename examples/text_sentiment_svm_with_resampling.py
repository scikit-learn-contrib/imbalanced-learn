#!/usr/bin/env python
"""
Text sentiment classification (negative/neutral/positive) with Linear SVM
and class imbalance handling using imbalanced-learn.

Dataset: tweet_eval/sentiment (3 classes)
Requires: pip install datasets matplotlib

Run:
    python examples/text_sentiment_svm_with_resampling.py --plot --max-samples 6000
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.svm import LinearSVC

try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This example requires the 'datasets' package.\n"
        "Install it with:\n    pip install datasets\n"
    ) from e


def load_tweet_eval(max_samples: int | None = 6000, random_state: int = 42):
    """Load 3-class sentiment from tweet_eval.

    Returns X_train, y_train, X_test, y_test.
    If max_samples is set, subsamples training data for speed.
    """
    ds = load_dataset("tweet_eval", "sentiment")
    # labels: 0=negative, 1=neutral, 2=positive
    def xy(split):
        X = [ex["text"] for ex in split]
        y = np.array([ex["label"] for ex in split], dtype=int)
        return X, y

    X_tr, y_tr = xy(ds["train"])
    X_va, y_va = xy(ds["validation"])
    X_te, y_te = xy(ds["test"])

    # merge train+validation for a larger training pool
    X_train = X_tr + X_va
    y_train = np.concatenate([y_tr, y_va])

    if max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_train), size=min(max_samples, len(X_train)), replace=False)
        X_train = [X_train[i] for i in idx]
        y_train = y_train[idx]
        # also downsample test a bit for quick runs
        idx_t = rng.choice(len(X_te), size=min(max_samples // 3 + 300, len(X_te)), replace=False)
        X_test = [X_te[i] for i in idx_t]
        y_test = y_te[idx_t]
    else:
        X_test, y_test = X_te, y_te

    return X_train, y_train, X_test, y_test


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="3-class sentiment with LinearSVC and RandomUnderSampler."
    )
    parser.add_argument("--max-samples", type=int, default=6000,
                        help="Max training samples for speed (set None for full).")
    parser.add_argument("--plot", action="store_true", help="Save confusion matrix PNG.")
    parser.add_argument("--output", type=str, default="confmat_svm_imblearn.png",
                        help="Output path for confusion matrix.")
    args = parser.parse_args(argv)

    X_train, y_train, X_test, y_test = load_tweet_eval(max_samples=args.max_samples)

    # Note: SMOTE does not support sparse input; use an under-sampler for text
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
        ("balance", RandomUnderSampler(random_state=0)),
        ("clf", LinearSVC()),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced accuracy: {bal_acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))

    if args.plot:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        ConfusionMatrixDisplay(cm, display_labels=["neg", "neu", "pos"]).plot(values_format="d")
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Saved confusion matrix to {args.output}")


if __name__ == "__main__":
    main()
