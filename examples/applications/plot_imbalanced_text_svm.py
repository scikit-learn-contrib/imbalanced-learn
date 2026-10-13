"""
========================================================
Linear SVM for imbalanced three-class sentiment analysis
========================================================

This example trains a linear support vector machine on TF-IDF features to
predict whether a tweet is negative, neutral or positive. Negative tweets
dominate the data, so we compare two ways of reducing the resulting bias:
re-weighting the classes inside the SVM and over-sampling the minority classes.
"""

# Authors: Aryan Mittal <aryanmittal168@gmail.com>
# License: MIT

# %%
print(__doc__)

# %% [markdown]
# Discussion
# Setting the data set
# --------------------
#
# We use the "Twitter US Airline Sentiment" data set, fetched from OpenML. It
# contains tweets about US airlines labeled as negative, neutral or positive.
# The classes are naturally imbalanced. We keep 30% of the data for testing and
# stratify the split so that both sets keep the same class proportions.

# %%
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

df = fetch_openml(data_id=43628, as_frame=True).frame
X = df["text"]  # check
y = df["airline_sentiment"]  # check

labels = ["negative", "neutral", "positive"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)
print(y_train.value_counts())

# %% [markdown]
# Three strategies
# ----------------
#
# 1. a plain :class:`~sklearn.svm.LinearSVC`;
# 2. the same SVM with ``class_weight="balanced"``;
# 3. :class:`~imblearn.over_sampling.RandomOverSampler` followed by the SVM.
#    The sampler must be used inside an :class:`~imblearn.pipeline.Pipeline`
#    from imbalanced-learn so that resampling only happens during ``fit``.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline


def make_vectorizer():
    return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)


models = {
    "baseline": make_pipeline(make_vectorizer(), LinearSVC()),
    "class_weight": make_pipeline(
        make_vectorizer(), LinearSVC(class_weight="balanced")
    ),
    "oversampling": make_pipeline(
        make_vectorizer(), RandomOverSampler(random_state=0), LinearSVC()
    ),
}

predictions = {
    name: model.fit(X_train, y_train).predict(X_test) for name, model in models.items()
}

# %% [markdown]
# Metrics on the test set
# -----------------------
#
# Accuracy mostly reflects the majority class, so we report balanced accuracy
# and macro F1 as well.

# %%
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

for name, y_pred in predictions.items():
    print(
        f"{name:>13}: accuracy = {accuracy_score(y_test, y_pred):.3f}, "
        f"balanced accuracy = {balanced_accuracy_score(y_test, y_pred):.3f}, "
        f"macro F1 = {f1_score(y_test, y_pred, average='macro'):.3f}"
    )

# %%
from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, predictions["baseline"]))

# %% [markdown]
# Per-class recall
# ----------------
#
# The recall of the minority classes is where the strategies differ most.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score

fig, ax = plt.subplots(figsize=(8, 4))
width = 0.25
x = np.arange(len(labels))
for i, (name, y_pred) in enumerate(predictions.items()):
    recalls = recall_score(y_test, y_pred, labels=labels, average=None)
    ax.bar(x + i * width, recalls, width, label=name)
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylabel("Recall")
ax.set_title("Per-class recall on the test set")
ax.legend()
fig.tight_layout()
plt.show()
