from numpy.testing import assert_allclose

import sklearn

from sklearn.datasets import make_blobs
from sklearn.metrics import make_scorer
from sklearn.svm import LinearSVC

from imblearn.metrics import (sensitivity_score, specificity_score,
                              geometric_mean_score,
                              make_index_balanced_accuracy)
# Get the version
sk_version = sklearn.__version__
if sk_version < '0.18':
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import train_test_split, GridSearchCV

R_TOL = 1e-2


def test_imblearn_classification_scorers():
    """Test if the implemented scorer can be used in scikit-learn"""
    X, y = make_blobs(random_state=0, centers=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LinearSVC(random_state=0)
    clf.fit(X_train, y_train)

    # sensitivity scorer
    scorer = make_scorer(sensitivity_score, pos_label=None, average='macro')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    scorer = make_scorer(sensitivity_score, pos_label=None, average='weighted')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    scorer = make_scorer(sensitivity_score, pos_label=None, average='micro')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    scorer = make_scorer(sensitivity_score, pos_label=1)
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    # specificity scorer
    scorer = make_scorer(specificity_score, pos_label=None, average='macro')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    scorer = make_scorer(specificity_score, pos_label=None, average='weighted')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    scorer = make_scorer(specificity_score, pos_label=None, average='micro')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    scorer = make_scorer(specificity_score, pos_label=1)
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.95, rtol=R_TOL)

    # geometric_mean scorer
    scorer = make_scorer(geometric_mean_score, pos_label=None, average='macro')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    scorer = make_scorer(
        geometric_mean_score, pos_label=None, average='weighted')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    scorer = make_scorer(geometric_mean_score, pos_label=None, average='micro')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    scorer = make_scorer(geometric_mean_score, pos_label=1)
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.92, rtol=R_TOL)

    # make a iba metric before a scorer
    geo_mean_iba = make_index_balanced_accuracy()(geometric_mean_score)
    scorer = make_scorer(geo_mean_iba, pos_label=None, average='macro')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.85, rtol=R_TOL)

    scorer = make_scorer(geo_mean_iba, pos_label=None, average='weighted')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.85, rtol=R_TOL)

    scorer = make_scorer(geo_mean_iba, pos_label=None, average='micro')
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.85, rtol=R_TOL)

    scorer = make_scorer(geo_mean_iba, pos_label=1)
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=scorer)
    grid.fit(X_train, y_train).predict(X_test)
    assert_allclose(grid.best_score_, 0.84, rtol=R_TOL)
