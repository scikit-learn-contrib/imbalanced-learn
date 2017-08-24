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

from collections import Counter

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

print(__doc__)

###############################################################################
# Setting the data set
###############################################################################

###############################################################################
# We use a part of the 20 newsgroups data set by loading 4 topics. Using the
# scikit-learn loader, the data are split into a training and a testing set.
#
# Note the class \#3 is the minority class and has almost twice less samples
# than the majority class.

categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     categories=categories)

X_train = newsgroups_train.data
X_test = newsgroups_test.data

y_train = newsgroups_train.target
y_test = newsgroups_test.target

print('Training class distributions summary: {}'.format(Counter(y_train)))
print('Test class distributions summary: {}'.format(Counter(y_test)))

###############################################################################
# The usual scikit-learn pipeline
###############################################################################

###############################################################################
# You might usually use scikit-learn pipeline by combining the TF-IDF
# vectorizer to feed a multinomial naive bayes classifier. A classification
# report summarized the results on the testing set.
#
# As expected, the recall of the class \#3 is low mainly due to the class
# imbalanced.

pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(classification_report_imbalanced(y_test, y_pred))

###############################################################################
# Balancing the class before classification
###############################################################################

###############################################################################
# To improve the prediction of the class \#3, it could be interesting to apply
# a balancing before to train the naive bayes classifier. Therefore, we will
# use a ``RandomUnderSampler`` to equalize the number of samples in all the
# classes before the training.
#
# It is also important to note that we are using the ``make_pipeline`` function
# implemented in imbalanced-learn to properly handle the samplers.

pipe = make_pipeline_imb(TfidfVectorizer(),
                         RandomUnderSampler(),
                         MultinomialNB())

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

###############################################################################
# Although the results are almost identical, it can be seen that the resampling
# allowed to correct the poor recall of the class \#3 at the cost of reducing
# the other metrics for the other classes. However, the overall results are
# slightly better.

print(classification_report_imbalanced(y_test, y_pred))
