"""
=============================================
Compare ensemble classifiers using resampling
=============================================

Ensemble classifiers have shown to improve classification performance compare
to single learner. However, they will be affected by class imbalance. This
example shows the benefit of balancing the training set before to learn
learners. We are making the comparison with non-balanced ensemble methods.

We make a comparison using `skore.evaluate` to obtain a structured report
of the different classifiers.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

# %%
print(__doc__)

# %% [markdown]
# Load an imbalanced dataset
# --------------------------
#
# We will load the UCI SatImage dataset which has an imbalanced ratio of 9.3:1
# (number of majority sample for a minority sample). The data are then split
# into training and testing.

from sklearn.model_selection import train_test_split

# %%
from imblearn.datasets import fetch_datasets

satimage = fetch_datasets()["satimage"]
X, y = satimage.data, satimage.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# %% [markdown]
# Classification using a single decision tree
# -------------------------------------------
#
# We train a decision tree classifier which will be used as a baseline for the
# rest of this example.
#
# The results are reported using `skore.evaluate` which provides a structured
# report of the classifier performance.

# %%
import skore
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

report_tree = skore.evaluate(tree, X_test, y_test, splitter="prefit")
report_tree.metrics.summarize().frame()

# %%
report_tree.metrics.confusion_matrix().plot()

# %% [markdown]
# Classification using bagging classifier with and without sampling
# -----------------------------------------------------------------
#
# Instead of using a single tree, we will check if an ensemble of decision tree
# can actually alleviate the issue induced by the class imbalancing. First, we
# will use a bagging classifier and its counter part which internally uses a
# random under-sampling to balanced each bootstrap sample.

# %%
from sklearn.ensemble import BaggingClassifier

from imblearn.ensemble import BalancedBaggingClassifier

bagging = BaggingClassifier(n_estimators=50, random_state=0)
balanced_bagging = BalancedBaggingClassifier(n_estimators=50, random_state=0)

bagging.fit(X_train, y_train)
balanced_bagging.fit(X_train, y_train)

# %% [markdown]
# Balancing each bootstrap sample allows to increase significantly the balanced
# accuracy and the geometric mean.

# %%
report_bagging = skore.evaluate(bagging, X_test, y_test, splitter="prefit")
report_bagging.metrics.summarize().frame()

# %%
report_balanced_bagging = skore.evaluate(
    balanced_bagging, X_test, y_test, splitter="prefit"
)
report_balanced_bagging.metrics.summarize().frame()

# %%
report_bagging.metrics.confusion_matrix().plot()

# %%
report_balanced_bagging.metrics.confusion_matrix().plot()

# %% [markdown]
# Classification using random forest classifier with and without sampling
# -----------------------------------------------------------------------
#
# Random forest is another popular ensemble method and it is usually
# outperforming bagging. Here, we used a vanilla random forest and its balanced
# counterpart in which each bootstrap sample is balanced.

# %%
from sklearn.ensemble import RandomForestClassifier

from imblearn.ensemble import BalancedRandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, random_state=0)
brf = BalancedRandomForestClassifier(
    n_estimators=50,
    sampling_strategy="all",
    replacement=True,
    bootstrap=False,
    random_state=0,
)

rf.fit(X_train, y_train)
brf.fit(X_train, y_train)

# %% [markdown]
# Similarly to the previous experiment, the balanced classifier outperform the
# classifier which learn from imbalanced bootstrap samples. In addition, random
# forest outperforms the bagging classifier.

# %%
report_rf = skore.evaluate(rf, X_test, y_test, splitter="prefit")
report_rf.metrics.summarize().frame()

# %%
report_brf = skore.evaluate(brf, X_test, y_test, splitter="prefit")
report_brf.metrics.summarize().frame()

# %%
report_rf.metrics.confusion_matrix().plot()

# %%
report_brf.metrics.confusion_matrix().plot()

# %% [markdown]
# Boosting classifier
# -------------------
#
# In the same manner, easy ensemble classifier is a bag of balanced AdaBoost
# classifier. However, it will be slower to train than random forest and will
# achieve worse performance.

# %%
from sklearn.ensemble import AdaBoostClassifier

from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier

estimator = AdaBoostClassifier(n_estimators=10)
eec = EasyEnsembleClassifier(n_estimators=10, estimator=estimator)
eec.fit(X_train, y_train)

rusboost = RUSBoostClassifier(n_estimators=10, estimator=estimator)
rusboost.fit(X_train, y_train)

# %%
report_eec = skore.evaluate(eec, X_test, y_test, splitter="prefit")
report_eec.metrics.summarize().frame()

# %%
report_rusboost = skore.evaluate(rusboost, X_test, y_test, splitter="prefit")
report_rusboost.metrics.summarize().frame()

# %%
report_eec.metrics.confusion_matrix().plot()

# %%
import matplotlib.pyplot as plt

report_rusboost.metrics.confusion_matrix().plot()
plt.show()
