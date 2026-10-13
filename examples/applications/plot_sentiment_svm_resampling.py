"""
================================================================
Three-class tweet sentiment with a linear SVM and resampling
================================================================

This example classifies short texts as **negative**, **neutral** or
**positive** with a linear Support Vector Machine (SVM) trained on TF-IDF
features. It uses the sentiment task of the TweetEval benchmark
(Barbieri et al., 2020), which is built from SemEval-2017 Task 4.

Real sentiment data is rarely balanced. In TweetEval, negative tweets are the
smallest class, and in many production settings (complaint detection, for
example) they are even rarer. A plain SVM then favours the majority classes
and misses the tweets we care about most.

We therefore compare four ways of training the same classifier:

* no correction (the usual scikit-learn pipeline),
* cost-sensitive learning with ``class_weight="balanced"``,
* random under-sampling inside an imbalanced-learn
  :class:`~imblearn.pipeline.Pipeline`,
* random over-sampling inside the same kind of pipeline.

The comparison is run twice: on the original class distribution and on a
training set where we artificially keep only 10% of the negative tweets with
:func:`~imblearn.datasets.make_imbalance`. Resamplers in an imbalanced-learn
pipeline are only applied during ``fit``, so the test set is never resampled.
"""

# Authors: Mansi Verma
# License: MIT

# %%
print(__doc__)

import os
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import get_data_home
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.svm import LinearSVC

from imblearn.datasets import make_imbalance
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

RANDOM_STATE = 0
LABEL_NAMES = np.array(["negative", "neutral", "positive"])
TWEETEVAL_URL = (
    "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/"
    "datasets/sentiment/{subset}_{kind}.txt"
)


# %% [markdown]
# Loading the data
# ----------------
#
# TweetEval ships one tweet per line in ``{subset}_text.txt`` and the matching
# integer label (0 = negative, 1 = neutral, 2 = positive) in
# ``{subset}_labels.txt``. The helper below downloads both files once and
# caches them in the scikit-learn data folder, so later runs work offline.


# %%
def fetch_tweeteval_sentiment(subset="train", data_home=None):
    """Load the TweetEval sentiment split, downloading it on first use.

    Parameters
    ----------
    subset : {"train", "val", "test"}, default="train"
        Which split to load.
    data_home : str or path-like, default=None
        Cache folder. Defaults to ``sklearn.datasets.get_data_home()``.

    Returns
    -------
    texts : list of str
        One tweet per element.
    labels : ndarray of shape (n_samples,)
        Sentiment labels encoded as 0 (negative), 1 (neutral), 2 (positive).
    """
    if subset not in {"train", "val", "test"}:
        raise ValueError(f"subset must be 'train', 'val' or 'test', got {subset!r}")
    folder = Path(get_data_home(data_home)) / "tweeteval_sentiment"
    folder.mkdir(parents=True, exist_ok=True)

    paths = {}
    for kind in ("text", "labels"):
        path = folder / f"{subset}_{kind}.txt"
        if not path.exists():
            url = TWEETEVAL_URL.format(subset=subset, kind=kind)
            with urlopen(url) as response:  # noqa: S310 (fixed https URL)
                path.write_bytes(response.read())
        paths[kind] = path

    texts = paths["text"].read_text(encoding="utf-8").splitlines()
    labels = np.atleast_1d(np.loadtxt(paths["labels"], dtype=int))
    if len(texts) != len(labels):
        raise ValueError(
            f"{subset}: {len(texts)} texts but {len(labels)} labels; "
            "delete the cached files and download again."
        )
    return texts, labels


X_train, y_train = fetch_tweeteval_sentiment("train")
X_test, y_test = fetch_tweeteval_sentiment("test")

print(f"Training distribution: {Counter(LABEL_NAMES[y_train].tolist())}")
print(f"Testing distribution:  {Counter(LABEL_NAMES[y_test].tolist())}")

# %% [markdown]
# Building the SVM pipeline
# -------------------------
#
# ``LinearSVC`` solves the same linear-kernel SVM problem as
# ``SVC(kernel="linear")`` but scales to tens of thousands of documents.
# Word unigrams and bigrams keep short negations such as "not good". An
# optional sampler sits between the vectorizer and the SVM, which is only
# possible with :func:`imblearn.pipeline.make_pipeline`: the sampler runs on
# the TF-IDF matrix during ``fit`` and is skipped during ``predict``.


# %%
def make_sentiment_pipeline(sampler=None, class_weight=None, C=0.5):
    """Create a TF-IDF + linear SVM pipeline with an optional sampler.

    Parameters
    ----------
    sampler : imbalanced-learn sampler or None, default=None
        Resampling step applied to the TF-IDF features during ``fit`` only.
    class_weight : dict, "balanced" or None, default=None
        Passed to :class:`~sklearn.svm.LinearSVC`.
    C : float, default=0.5
        Inverse regularization strength of the SVM.

    Returns
    -------
    pipeline : imblearn.pipeline.Pipeline
        Unfitted pipeline.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True
    )
    svm = LinearSVC(C=C, class_weight=class_weight, random_state=RANDOM_STATE)
    steps = [vectorizer] + ([sampler] if sampler is not None else []) + [svm]
    return make_pipeline(*steps)


def make_strategies():
    """Return the four training strategies compared in this example."""
    return {
        "No correction": make_sentiment_pipeline(),
        "class_weight='balanced'": make_sentiment_pipeline(class_weight="balanced"),
        "Random under-sampling": make_sentiment_pipeline(
            sampler=RandomUnderSampler(random_state=RANDOM_STATE)
        ),
        "Random over-sampling": make_sentiment_pipeline(
            sampler=RandomOverSampler(random_state=RANDOM_STATE)
        ),
    }


def evaluate(pipeline, X_tr, y_tr, X_te, y_te):
    """Fit ``pipeline`` and return imbalance-aware scores on the test set."""
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    recalls = recall_score(y_te, y_pred, average=None, labels=[0, 1, 2])
    return {
        "macro F1": f1_score(y_te, y_pred, average="macro"),
        "balanced accuracy": balanced_accuracy_score(y_te, y_pred),
        "geometric mean": geometric_mean_score(y_te, y_pred, average="multiclass"),
        "recall negative": recalls[0],
        "recall neutral": recalls[1],
        "recall positive": recalls[2],
    }


# %% [markdown]
# Scenario 1: the original class distribution
# -------------------------------------------

# %%
results_original = pd.DataFrame(
    {
        name: evaluate(pipe, X_train, y_train, X_test, y_test)
        for name, pipe in make_strategies().items()
    }
).T
print(results_original.round(3).to_string())

# %% [markdown]
# Scenario 2: negative tweets become rare
# ---------------------------------------
#
# We keep only 10% of the negative training tweets. ``make_imbalance`` works on
# 2-D inputs, so we pass row indices and select the texts afterwards. The test
# set is left untouched.

# %%
n_negative = int(np.sum(y_train == 0) * 0.10)
idx_rare, y_train_rare = make_imbalance(
    np.arange(len(y_train)).reshape(-1, 1),
    y_train,
    sampling_strategy={0: n_negative},
    random_state=RANDOM_STATE,
)
X_train_rare = [X_train[i] for i in idx_rare.ravel()]
rare_counts = Counter(LABEL_NAMES[y_train_rare].tolist())
print(f"Rare-negative training distribution: {rare_counts}")

results_rare = pd.DataFrame(
    {
        name: evaluate(pipe, X_train_rare, y_train_rare, X_test, y_test)
        for name, pipe in make_strategies().items()
    }
).T
print(results_rare.round(3).to_string())

# %% [markdown]
# Comparing the strategies
# ------------------------
#
# Overall scores hide what happens to each class, so we plot the recall of
# the negative class next to macro F1 for both scenarios.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (title, res) in zip(
    axes,
    [("Original distribution", results_original), ("10% of negatives", results_rare)],
):
    res[["recall negative", "macro F1"]].plot.barh(ax=ax, color=["#c0504d", "#4f81bd"])
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score on the untouched test set")
    ax.legend(loc="lower right")
axes[0].invert_yaxis()
fig.suptitle("Linear SVM on TweetEval sentiment")
fig.tight_layout()

# %% [markdown]
# A detailed report for the best rare-negative strategy
# -----------------------------------------------------
#
# :func:`~imblearn.metrics.classification_report_imbalanced` adds
# specificity and the geometric mean per class to the usual report.

# %%
best_name = results_rare["macro F1"].idxmax()
best_pipeline = make_strategies()[best_name].fit(X_train_rare, y_train_rare)
print(f"Best strategy with rare negatives: {best_name}")
print(
    classification_report_imbalanced(
        y_test, best_pipeline.predict(X_test), target_names=list(LABEL_NAMES)
    )
)

# %% [markdown]
# Predicting the sentiment of new text
# ------------------------------------

# %%
new_texts = [
    "The new update is fantastic, battery life is so much better!",
    "The store opens at 9am on Saturday.",
    "Worst customer service ever, still waiting for my refund.",
]
for text, label in zip(new_texts, best_pipeline.predict(new_texts)):
    print(f"{LABEL_NAMES[label]:>8}  <-  {text}")

# %% [markdown]
# Take-home messages
# ------------------
#
# * With the original distribution, every correction raises the recall of the
#   negative class (about 0.42 without correction). ``class_weight="balanced"``
#   gives the best macro F1, and under-sampling gives the highest negative
#   recall at the cost of neutral recall.
# * When negatives are rare, the uncorrected SVM almost never predicts them.
#   Cost weighting and over-sampling help only a little in this setting:
#   repeating or up-weighting 709 tweets adds no new vocabulary, and the
#   one-vs-rest scores of the other two classes still dominate. Under-sampling
#   removes most of the majority tweets the SVM sees and recovers more than
#   half of the negative tweets.
# * The best strategy depends on how imbalanced the data is, so compare
#   strategies on your own data with imbalance-aware metrics (macro F1,
#   balanced accuracy, geometric mean) rather than plain accuracy.
# * Because the samplers live inside the imbalanced-learn pipeline, they are
#   never applied to the test data, which avoids an easy source of leakage.
#
# Reference: F. Barbieri, J. Camacho-Collados, L. Espinosa-Anke and
# L. Neves, "TweetEval: Unified Benchmark and Comparative Evaluation for
# Tweet Classification", Findings of EMNLP, 2020.

# %%
if os.environ.get("SHOW_PLOTS", "1") == "1":
    plt.show()
