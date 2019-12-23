import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from imblearn.tree.criterion import HellingerDistanceCriterion

X, y = make_classification(
    n_samples=10000, n_features=40, n_informative=5,
    n_classes=2, weights=[0.05, 0.95], random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

hdc = HellingerDistanceCriterion(1, np.array([2], dtype='int64'))
clf = RandomForestClassifier(criterion=hdc, max_depth=4, n_estimators=100)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
