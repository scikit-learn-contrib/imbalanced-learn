"""
=================================================
Example of topic classification in text documents
=================================================

This example shows how to balance the text data before to train a classifier.

Note that for this example, the data are slightly imbalanced but it can happen
that for some data sets, the imbalanced ratio is more significant.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

# %%
print(__doc__)

# %% [markdown]
# Setting the data set
# --------------------
#
# We use a part of the 20 newsgroups data set by loading 4 topics. Using the
# scikit-learn loader, the data are split into a training and a testing set.
#
# Note the class \#3 is the minority class and has almost twice less samples
# than the majority class.

# %%
from sklearn.datasets import fetch_20newsgroups

categories = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]
newsgroups_train = fetch_20newsgroups(subset="train", categories=categories)
newsgroups_test = fetch_20newsgroups(subset="test", categories=categories)

X_train = newsgroups_train.data
X_test = newsgroups_test.data

y_train = newsgroups_train.target
y_test = newsgroups_test.target

# %%
from collections import Counter

print(f"Training class distributions summary: {Counter(y_train)}")
print(f"Test class distributions summary: {Counter(y_test)}")

# %% [markdown]
# The usual scikit-learn pipeline
# -------------------------------
#
# You might usually use scikit-learn pipeline by combining the TF-IDF
# vectorizer to feed a multinomial naive bayes classifier. A classification
# report summarized the results on the testing set.
#
# As expected, the recall of the class \#3 is low mainly due to the class
# imbalanced.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))

# %% [markdown]
# Balancing the class before classification
# -----------------------------------------
#
# To improve the prediction of the class \#3, it could be interesting to apply
# a balancing before to train the naive bayes classifier. Therefore, we will
# use a :class:`~imblearn.under_sampling.RandomUnderSampler` to equalize the
# number of samples in all the classes before the training.
#
# It is also important to note that we are using the
# :class:`~imblearn.pipeline.make_pipeline` function implemented in
# imbalanced-learn to properly handle the samplers.

from imblearn.pipeline import make_pipeline as make_pipeline_imb

# %%
from imblearn.under_sampling import RandomUnderSampler

model = make_pipeline_imb(TfidfVectorizer(), RandomUnderSampler(), MultinomialNB())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %% [markdown]
# Although the results are almost identical, it can be seen that the resampling
# allowed to correct the poor recall of the class \#3 at the cost of reducing
# the other metrics for the other classes. However, the overall results are
# slightly better.

# %%
print(classification_report_imbalanced(y_test, y_pred))
