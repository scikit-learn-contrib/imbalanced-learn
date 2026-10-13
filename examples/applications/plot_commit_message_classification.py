"""
===================================================
Classifying commit messages with imbalanced classes
===================================================

Commit messages skew heavily toward bug fixes: in a typical history, most
commits fix something, fewer add a feature, and only a handful do anything
else. A classifier trained on such data learns that predicting "fix" is
usually right, and its overall accuracy hides a poor recall on the minority
classes.

This example classifies commit messages into "fix", "feature" and "other"
using TF-IDF features and a linear SVM. It compares a scikit-learn pipeline
with an imbalanced-learn pipeline that over-samples the training set, and
reports the per-class recall of both.
"""

# Authors: Srija Taduri <srija.taduri1998@gmail.com>
# License: MIT

# %%
print(__doc__)

RANDOM_STATE = 0

# %% [markdown]
# Generating the commit messages
# ------------------------------
#
# We generate synthetic commit messages from templates so that the example is
# reproducible and does not need any download. Each message has a subject line
# and an optional body. Some subjects are explicit ("Fix memory leak in
# cache") while others are vague ("Update parser") and only the body tells
# what the commit is about ("Fixes #123."), as often happens in real
# histories.
#
# The classes are drawn with a probability of 70% for "fix", 20% for
# "feature" and 10% for "other".

# %%
import numpy as np

MODULES = [
    "parser",
    "cli",
    "config loader",
    "cache",
    "logger",
    "scheduler",
    "http client",
    "exporter",
]
PROBLEMS = [
    "off-by-one error",
    "race condition",
    "memory leak",
    "wrong default value",
    "timeout",
    "unicode handling",
    "rounding error",
]
CAPABILITIES = [
    "a --verbose option",
    "a retry policy",
    "a dry-run mode",
    "streaming output",
    "plugin hooks",
    "yaml files",
    "gzip input",
]

SUBJECTS = {
    "fix": [
        "Fix {problem} in {module}",
        "Fix {module} {problem}",
        "Bugfix: {problem} in {module}",
        "Fix crash when {module} gets empty input",
    ],
    "feature": [
        "Add {capability} to {module}",
        "Implement {capability} in {module}",
        "Support {capability} in {module}",
        "Introduce {capability} for {module}",
    ],
    "other": [
        "Update docs for {module}",
        "Refactor {module}",
        "Bump version to 0.{n}.0",
        "Update changelog",
        "Reformat {module} with black",
        "Rename variables in {module}",
    ],
}
# Vague subjects are shared by all classes: only the body carries the intent.
VAGUE_SUBJECTS = [
    "Update {module}",
    "Improve {module} handling",
    "Tweak {module} defaults",
    "Rework {module} internals",
]
BODIES = {
    "fix": [
        "Fixes #{n}.",
        "This fixes a bug reported by users.",
        "Regression introduced in #{n}.",
    ],
    "feature": [
        "This adds {capability}.",
        "Implements #{n}.",
        "Users can now opt in to {capability}.",
    ],
    "other": [
        "No functional change.",
        "See https://example.com/docs/{n} for details.",
        "",
    ],
}
# Probability that a commit of a given class only has a vague subject.
VAGUE_RATE = {"fix": 0.25, "feature": 0.4, "other": 0.5}


def make_commit_messages(n_samples, weights, random_state=None):
    """Generate raw commit messages from templates."""
    rng = np.random.default_rng(random_state)
    classes = rng.choice(list(weights), size=n_samples, p=list(weights.values()))
    messages = []
    for klass in classes:
        if rng.random() < VAGUE_RATE[klass]:
            subject = VAGUE_SUBJECTS[rng.integers(len(VAGUE_SUBJECTS))]
            # skip the last body of each class: for "feature" it has no keyword
            # and the commit would be labeled "other" instead of "feature"
            body = BODIES[klass][rng.integers(len(BODIES[klass]) - 1)]
        else:
            subject = SUBJECTS[klass][rng.integers(len(SUBJECTS[klass]))]
            body = BODIES[klass][rng.integers(len(BODIES[klass]))]
        if rng.random() < 0.3:
            # squash-merged pull requests reference the PR in the subject
            subject += " (#{n})"
        message = f"{subject}\n\n{body}".format(
            module=MODULES[rng.integers(len(MODULES))],
            problem=PROBLEMS[rng.integers(len(PROBLEMS))],
            capability=CAPABILITIES[rng.integers(len(CAPABILITIES))],
            n=rng.integers(1, 2000),
        )
        messages.append(message.strip())
    return messages


messages = make_commit_messages(
    n_samples=1_000,
    weights={"fix": 0.7, "feature": 0.2, "other": 0.1},
    random_state=RANDOM_STATE,
)
print(messages[0])

# %% [markdown]
# Labeling the commits
# --------------------
#
# Real commit histories do not come with labels. A common approach is to
# derive them from keywords found anywhere in the message, including the body.
# A commit mentioning a bug is labeled "fix" even if it also adds something
# (e.g. "Add regression test for bug #12"), so the "fix" keywords take
# precedence over the "feature" ones.

# %%
import re

FIX_KEYWORDS = {
    "fix",
    "fixes",
    "fixed",
    "bug",
    "bugs",
    "bugfix",
    "crash",
    "crashes",
    "regression",
    "hotfix",
}
FEATURE_KEYWORDS = {
    "add",
    "adds",
    "added",
    "feature",
    "implement",
    "implements",
    "implemented",
    "introduce",
    "introduces",
    "support",
    "supports",
}


def label_commit(message):
    """Label a commit message as "fix", "feature" or "other"."""
    if not message:
        return "other"
    words = set(re.findall(r"[a-z]+", message.lower()))
    if words & FIX_KEYWORDS:
        return "fix"
    if words & FEATURE_KEYWORDS:
        return "feature"
    return "other"


# %%
from collections import Counter

y = np.array([label_commit(message) for message in messages])
print(f"Class distribution: {Counter(y.tolist())}")

# %% [markdown]
# Cleaning the messages
# ---------------------
#
# The classifier only sees the subject line, as a short summary shown by
# ``git log --oneline`` would. We lowercase it and strip URLs and issue
# references such as ``#123`` which carry no information about the class.

# %%


def clean_message(message):
    """Keep the lowercased subject line without URLs and issue references."""
    if not message:
        return ""
    lines = message.strip().splitlines()
    subject = lines[0].lower() if lines else ""
    subject = re.sub(r"(https?://|www\.)\S+", " ", subject)
    subject = re.sub(r"\(?#\d+\)?", " ", subject)
    return " ".join(subject.split())


X = np.array([clean_message(message) for message in messages])

raw = "Fix memory leak in cache (#42)\n\nSee https://example.com/issues/42"
print(f"Before: {raw!r}")
print(f"After:  {clean_message(raw)!r}")

# %% [markdown]
# We keep the class proportions in the training and testing sets by
# stratifying the split.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=RANDOM_STATE
)

# %% [markdown]
# The usual scikit-learn pipeline
# -------------------------------
#
# We chain a :class:`~sklearn.feature_extraction.text.TfidfVectorizer` with a
# :class:`~sklearn.svm.LinearSVC` in a scikit-learn
# :class:`~sklearn.pipeline.Pipeline`.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

baseline = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("svc", LinearSVC(random_state=RANDOM_STATE)),
    ]
)
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)
print(classification_report(y_test, y_pred_baseline, zero_division=0))

# %% [markdown]
# The vague subjects ("update parser") are mostly written by fixes, so the
# classifier predicts "fix" for them, which costs recall on the "feature" and
# "other" classes. The accuracy remains high because "fix" dominates the
# testing set.
#
# Over-sampling the training set
# ------------------------------
#
# We add a :class:`~imblearn.over_sampling.RandomOverSampler` between the
# vectorizer and the classifier. It duplicates samples of the minority classes
# until all classes have as many samples as the majority class.
#
# The sampler must only see the training data. Resampling before splitting
# would copy duplicates of the same commit in both the training and testing
# sets and inflate the scores. A scikit-learn
# :class:`~sklearn.pipeline.Pipeline` cannot hold a sampler, so we use the
# :class:`~imblearn.pipeline.Pipeline` of imbalanced-learn: it applies the
# sampler during ``fit`` only and skips it during ``predict``. Within a
# cross-validation, only the training folds are therefore resampled.

# %%
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

resampled = ImbPipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
        ("svc", LinearSVC(random_state=RANDOM_STATE)),
    ]
)
resampled.fit(X_train, y_train)
y_pred_resampled = resampled.predict(X_test)
print(classification_report(y_test, y_pred_resampled, zero_division=0))

# %% [markdown]
# Comparing the per-class recall
# ------------------------------

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score

labels = ["fix", "feature", "other"]
support = Counter(y_test)
models = {
    "scikit-learn Pipeline": y_pred_baseline,
    "imblearn Pipeline + RandomOverSampler": y_pred_resampled,
}

x = np.arange(len(labels))
width = 0.38
fig, ax = plt.subplots(figsize=(7, 4.5))
for offset, (name, y_pred) in zip([-width / 2, width / 2], models.items()):
    recall = recall_score(y_test, y_pred, labels=labels, average=None)
    accuracy = accuracy_score(y_test, y_pred)
    bars = ax.bar(x + offset, recall, width, label=f"{name} (accuracy {accuracy:.2f})")
    ax.bar_label(bars, fmt="%.2f", padding=2)

ax.set_xticks(x, [f"{label}\n({support[label]} test commits)" for label in labels])
ax.set_ylim(0, 1.4)
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_ylabel("Recall")
ax.set_title("Over-sampling trades recall on fixes for recall on minority classes")
ax.legend(loc="upper center")
fig.tight_layout()
plt.show()

# %% [markdown]
# Without resampling, the classifier reaches the highest accuracy but misses
# many of the "other" commits and a part of the "feature" commits.
# Over-sampling the training set raises the recall of both minority classes.
# In exchange, more fixes are predicted as one of the minority classes: the
# recall of "fix" and the overall accuracy decrease. Which model to prefer
# depends on the cost of missing a minority commit compared to mislabeling a
# fix.
#
# Limitations
# -----------
#
# This example should be read as an illustration rather than a benchmark:
#
# - The data are synthetic. Real commit messages are noisier, and labels
#   derived from keywords are themselves imperfect.
# - The messages come from a small set of templates. Their vocabulary makes
#   the explicit subjects almost perfectly separable, which inflates the
#   scores; the remaining errors come from the vague subjects only.
# - Over-sampling does not add information: it shifts the decision toward the
#   minority classes. Their recall increases at the cost of their precision,
#   and of the recall of the majority class.
