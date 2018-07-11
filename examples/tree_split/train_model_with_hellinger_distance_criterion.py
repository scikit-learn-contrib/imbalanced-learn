import numpy as np
import pandas as pd
from hellinger_distance_criterion import HellingerDistanceCriterion
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

# Random Forest criterions comparison
def compare_rf(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(criterion='gini', max_depth=4, n_estimators=100)
    clf.fit(X_train, y_train)
    print('gini score: ', clf.score(X_test, y_test))

    clf = RandomForestClassifier(criterion='entropy', max_depth=4, n_estimators=100)
    clf.fit(X_train, y_train)
    print('entropy score: ', clf.score(X_test, y_test))

    hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))
    clf = RandomForestClassifier(criterion=hdc, max_depth=4, n_estimators=100)
    clf.fit(X_train, y_train)
    print('hellinger distance score: ', clf.score(X_test, y_test))

# Decision Tree criterions comparison
def compare_dt(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=4)
    clf.fit(X_train, y_train)
    print('gini score: ', clf.score(X_test, y_test))

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    clf.fit(X_train, y_train)
    print('entropy score: ', clf.score(X_test, y_test))

    hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))
    clf = DecisionTreeClassifier(criterion=hdc, max_depth=4)
    clf.fit(X_train, y_train)
    print('hellinger distance score: ', clf.score(X_test, y_test))

# Comparison on breast cancer dataset
bc = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(bc.data, bc.target, test_size=0.3)
compare_rf(X_train, y_train, X_test, y_test)
compare_dt(X_train, y_train, X_test, y_test)

# Comparison on imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=40, n_informative=5, n_classes=2, weights=[0.05,0.95], random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
compare_rf(X_train, y_train, X_test, y_test)
compare_dt(X_train, y_train, X_test, y_test)
