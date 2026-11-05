import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import cross_validate
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn.model_selection import InstanceHardnessCV


@pytest.fixture
def data():
    return make_classification(
        weights=[0.5, 0.5],
        class_sep=0.5,
        n_informative=3,
        n_redundant=1,
        flip_y=0.05,
        n_samples=50,
        random_state=10,
    )


def test_groups_parameter_warning(data):
    """Test that a warning is raised when groups parameter is provided."""
    X, y = data
    ih_cv = InstanceHardnessCV(estimator=LogisticRegression(), n_splits=3)

    warning_msg = "The groups parameter is ignored by InstanceHardnessCV"
    with pytest.warns(UserWarning, match=warning_msg):
        list(ih_cv.split(X, y, groups=np.ones_like(y)))


def test_error_on_multiclass():
    """Test that an error is raised when the target is not binary."""
    X, y = make_classification(n_classes=3, n_clusters_per_class=1)
    err_msg = "InstanceHardnessCV only supports binary classification."
    with pytest.raises(ValueError, match=err_msg):
        next(InstanceHardnessCV(estimator=LogisticRegression()).split(X, y))


@pytest.mark.parametrize(
    "labels, pos_label",
    [
        ([0, 1], 2),
        ([-3, 7], 0),
        (["defaulted", "paid"], "missing"),
        ([False, True], "missing"),
        ([0, 1], "missing"),
        (["defaulted", "paid"], 1),
    ],
)
def test_invalid_pos_label(data, labels, pos_label):
    """An unknown positive label should identify the valid target labels."""
    X, y = data
    y = np.asarray(labels)[y]
    ih_cv = InstanceHardnessCV(
        estimator=LogisticRegression(), pos_label=pos_label, n_splits=3
    )

    with pytest.raises(ValueError, match="is not a valid label") as exc_info:
        next(ih_cv.split(X, y))

    expected_message = (
        f"pos_label={pos_label!r} is not a valid label. It should be one of {labels}."
    )
    assert str(exc_info.value) == expected_message


@pytest.mark.parametrize("labels", [[-3, 7], ["defaulted", "paid"], [False, True]])
@pytest.mark.parametrize("pos_label", [0, 1])
def test_valid_pos_label_preserves_splits(data, labels, pos_label):
    """Relabeling a binary target should preserve the train and test indices."""
    X, y = data
    labels = np.asarray(labels)
    original_cv = InstanceHardnessCV(
        estimator=LogisticRegression(), pos_label=pos_label, n_splits=3
    )
    relabeled_cv = InstanceHardnessCV(
        estimator=LogisticRegression(), pos_label=labels[pos_label], n_splits=3
    )

    for (train, test), (relabeled_train, relabeled_test) in zip(
        original_cv.split(X, y), relabeled_cv.split(X, labels[y])
    ):
        assert_array_equal(relabeled_train, train)
        assert_array_equal(relabeled_test, test)


def test_default_params(data):
    """Test that the default parameters are used."""
    X, y = data
    ih_cv = InstanceHardnessCV(estimator=LogisticRegression(), n_splits=3)
    cv_result = cross_validate(
        LogisticRegression(), X, y, cv=ih_cv, scoring="precision"
    )
    assert_allclose(cv_result["test_score"], [0.625, 0.6, 0.625], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype_target", [None, object])
def test_target_string_labels(data, dtype_target):
    """Test that the target can be a string array."""
    X, y = data
    labels = np.array(["a", "b"], dtype=dtype_target)
    y = labels[y]
    ih_cv = InstanceHardnessCV(estimator=LogisticRegression(), n_splits=3)
    cv_result = cross_validate(
        LogisticRegression(),
        X,
        y,
        cv=ih_cv,
        scoring=make_scorer(precision_score, pos_label="b"),
    )
    assert_allclose(cv_result["test_score"], [0.625, 0.6, 0.625], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype_target", [None, object])
def test_target_string_pos_label(data, dtype_target):
    """Test that the `pos_label` parameter can be used to select the positive class.

    Here, changing the `pos_label` will change the instance hardness and thus the
    `cv_result`.
    """
    X, y = data
    labels = np.array(["a", "b"], dtype=dtype_target)
    y = labels[y]
    ih_cv = InstanceHardnessCV(
        estimator=LogisticRegression(), pos_label="a", n_splits=3
    )
    cv_result = cross_validate(
        LogisticRegression(),
        X,
        y,
        cv=ih_cv,
        scoring=make_scorer(precision_score, pos_label="a"),
    )
    assert_allclose(
        cv_result["test_score"], [0.666667, 0.666667, 0.4], atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("n_splits", [2, 3, 4])
def test_n_splits(n_splits):
    """Test that the number of splits is correctly set."""
    ih_cv = InstanceHardnessCV(estimator=LogisticRegression(), n_splits=n_splits)
    assert ih_cv.get_n_splits() == n_splits
