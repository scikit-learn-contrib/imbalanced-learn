"""
=========================================================
Linear SVM text classification with an imbalanced dataset
=========================================================

A linear Support Vector Machine (SVM) on TF-IDF features is a strong baseline
for text classification, but when one topic is rare the SVM is biased towards
the frequent topic and misses many minority documents.

This example trains a :class:`~sklearn.svm.LinearSVC` on an imbalanced subset
of the 20 newsgroups data set (``sci.space`` vs. a rare ``sci.med``) and shows
how adding a :class:`~imblearn.under_sampling.RandomUnderSampler` to the
pipeline improves the recall of the minority topic.
"""

# Authors: The imbalanced-learn developers
# License: MIT

# %%
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from imblearn.datasets import make_imbalance
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

CATEGORIES = ["sci.med", "sci.space"]  # target 0 = sci.med, 1 = sci.space


def load_imbalanced_newsgroups(n_minority=60, random_state=0):
    """Load two newsgroups and keep only ``n_minority`` sci.med training texts.

    ``make_imbalance`` needs a 2D ``X``, so it is applied to the document
    indices, which are then used to select the texts. The test set is not
    changed.

    Returns
    -------
    X_train, y_train, X_test, y_test, target_names
    """
    remove = ("headers", "footers", "quotes")
    train = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=remove)
    test = fetch_20newsgroups(subset="test", categories=CATEGORIES, remove=remove)

    indices = np.arange(len(train.data)).reshape(-1, 1)
    indices, y_train = make_imbalance(
        indices,
        train.target,
        sampling_strategy={0: n_minority},
        random_state=random_state,
    )
    X_train = [train.data[i] for i in indices.ravel()]
    return X_train, y_train, test.data, test.target, train.target_names


def build_model(under_sample=False, random_state=0):
    """Return a TF-IDF + LinearSVC pipeline, optionally with under-sampling.

    The imbalanced-learn pipeline applies the sampler only during ``fit``.
    """
    steps = [TfidfVectorizer(stop_words="english", sublinear_tf=True)]
    if under_sample:
        steps.append(RandomUnderSampler(random_state=random_state))
    steps.append(LinearSVC())
    return make_pipeline(*steps)


# %%
X_train, y_train, X_test, y_test, target_names = load_imbalanced_newsgroups()
print(f"Training texts per class: {np.bincount(y_train)} {target_names}")

for under_sample in (False, True):
    model = build_model(under_sample=under_sample).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nWith" if under_sample else "\nWithout", "RandomUnderSampler:")
    print(classification_report_imbalanced(y_test, y_pred, target_names=target_names))

# %% [markdown]
# Without resampling, the recall ("rec") of the rare ``sci.med`` topic is low.
# With :class:`~imblearn.under_sampling.RandomUnderSampler` both topics have
# the same weight during training, which raises the minority recall.
