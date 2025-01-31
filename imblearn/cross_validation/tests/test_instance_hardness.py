import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.utils._testing import assert_almost_equal

from imblearn.cross_validation import InstanceHardnessCV

X, y = make_classification(
    weights=[0.9, 0.1],
    class_sep=2,
    n_informative=3,
    n_redundant=1,
    flip_y=0.05,
    n_samples=1000,
    random_state=10,
)


def test_instancehardness_cv():
    ih_cv = InstanceHardnessCV()
    clf = LogisticRegression(random_state=10)
    cv_result = cross_validate(clf, X, y, cv=ih_cv)
    assert_almost_equal(cv_result["test_score"].std(), 0.005, decimal=3)


@pytest.mark.parametrize("n_splits", [2, 3, 4])
def test_instancehardness_cv_n_splits(n_splits):
    ih_cv = InstanceHardnessCV(n_splits=n_splits, random_state=10)
    assert ih_cv.get_n_splits() == n_splits
