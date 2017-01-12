"""Test the module SMOTE ENN."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_equal, assert_raises, assert_warns)
from sklearn.utils.estimator_checks import check_estimator

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
              [1.34192108, -0.13367336], [0.62366841, -0.21312976],
              [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
              [0.74680821, 1.63827342], [0.2184254, 0.24299982],
              [0.61472253, -0.82309052], [0.19893132, -0.47761769],
              [1.06514042, -0.0770537], [0.97407872, 0.44454207],
              [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
              [-0.27410027, -0.54194484], [0.8381014, 0.44085498],
              [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
              [-0.00288378, 0.84259929], [1.79580611, -0.02219234]])
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4


def test_smote_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(SMOTETomek)


def test_smote_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    smote = SMOTETomek(ratio=ratio)
    assert_raises(ValueError, smote.fit, X, Y)

    # Define a ratio greater than 1
    ratio = 100.0
    smote = SMOTETomek(ratio=ratio)
    assert_raises(ValueError, smote.fit, X, Y)

    # Define ratio as an unknown string
    ratio = 'rnd'
    smote = SMOTETomek(ratio=ratio)
    assert_raises(ValueError, smote.fit, X, Y)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    smote = SMOTETomek(ratio=ratio)
    assert_raises(ValueError, smote.fit, X, Y)


def test_smote_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    smote = SMOTETomek(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, smote.fit, X, y_single_class)


def test_smote_fit():
    """Test the fitting method"""

    # Create the object
    smote = SMOTETomek(random_state=RND_SEED)
    # Fit the data
    smote.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(smote.min_c_, 0)
    assert_equal(smote.maj_c_, 1)
    assert_equal(smote.stats_c_[0], 8)
    assert_equal(smote.stats_c_[1], 12)


def test_smote_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    smote = SMOTETomek(random_state=RND_SEED)
    assert_raises(RuntimeError, smote.sample, X, Y)


def test_sample_regular():
    """Test sample function with regular SMOTE."""

    # Create the object
    smote = SMOTETomek(random_state=RND_SEED)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    X_gt = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769], [0.97407872, 0.44454207],
                     [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
                     [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
                     [-0.00288378, 0.84259929], [1.79580611, -0.02219234],
                     [0.38307743, -0.05670439], [0.93976473, -0.06570176],
                     [0.70319159, -0.02571668], [0.75052536, -0.19246517]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    """Test sample function with regular SMOTE and a ratio of 0.5."""

    # Create the object
    ratio = 0.8
    smote = SMOTETomek(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    X_gt = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769], [0.97407872, 0.44454207],
                     [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
                     [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
                     [-0.00288378, 0.84259929], [1.79580611, -0.02219234],
                     [0.45784496, -0.1053161]])
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    sm = SMOTETomek(random_state=RND_SEED)
    sm.fit(X, Y)
    assert_raises(RuntimeError, sm.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_multiclass_error():
    """ Test either if an error is raised when the target are not binary
    type. """

    # continuous case
    y = np.linspace(0, 1, 20)
    sm = SMOTETomek(random_state=RND_SEED)
    assert_warns(UserWarning, sm.fit, X, y)

    # multiclass case
    y = np.array([0] * 3 + [1] * 2 + [2] * 15)
    sm = SMOTETomek(random_state=RND_SEED)
    assert_warns(UserWarning, sm.fit, X, y)


def test_validate_estimator_init():
    """Test right processing while passing objects as initialization"""

    # Create a SMOTE and Tomek object
    smote = SMOTE(random_state=RND_SEED)
    tomek = TomekLinks(random_state=RND_SEED)

    smt = SMOTETomek(smote=smote, tomek=tomek, random_state=RND_SEED)

    X_resampled, y_resampled = smt.fit_sample(X, Y)

    X_gt = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769], [0.97407872, 0.44454207],
                     [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
                     [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
                     [-0.00288378, 0.84259929], [1.79580611, -0.02219234],
                     [0.38307743, -0.05670439], [0.93976473, -0.06570176],
                     [0.70319159, -0.02571668], [0.75052536, -0.19246517]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_default():
    """Test right processing while passing no object as initialization"""

    smt = SMOTETomek(random_state=RND_SEED)

    X_resampled, y_resampled = smt.fit_sample(X, Y)

    X_gt = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769], [0.97407872, 0.44454207],
                     [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
                     [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
                     [-0.00288378, 0.84259929], [1.79580611, -0.02219234],
                     [0.38307743, -0.05670439], [0.93976473, -0.06570176],
                     [0.70319159, -0.02571668], [0.75052536, -0.19246517]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_deprecation():
    """Test right processing while passing old parameters"""

    X_gt = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769], [0.97407872, 0.44454207],
                     [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
                     [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
                     [-0.00288378, 0.84259929], [1.79580611, -0.02219234],
                     [0.38307743, -0.05670439], [0.93976473, -0.06570176],
                     [0.70319159, -0.02571668], [0.75052536, -0.19246517]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])

    smt = SMOTETomek(random_state=RND_SEED, n_jobs=-1)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)

    smt = SMOTETomek(random_state=RND_SEED, k=5)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_error_wrong_object():
    """Test either if an error is raised while wrong objects are provided
    at the initialization"""

    # Create a SMOTE and Tomek object
    smote = 'rnd'
    tomek = 'rnd'

    smt = SMOTETomek(smote=smote, random_state=RND_SEED)
    assert_raises(ValueError, smt.fit, X, Y)
    smt = SMOTETomek(tomek=tomek, random_state=RND_SEED)
    assert_raises(ValueError, smt.fit, X, Y)
