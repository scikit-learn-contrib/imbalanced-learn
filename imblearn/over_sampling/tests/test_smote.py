"""Test the module SMOTE."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_warns

from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

from imblearn.over_sampling import SMOTE

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


def test_smote_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(SMOTE)


def test_smote_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    smote = SMOTE(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, smote.fit, X, Y)

    # Define a ratio greater than 1
    ratio = 100.0
    smote = SMOTE(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, smote.fit, X, Y)

    # Define ratio as an unknown string
    ratio = 'rnd'
    smote = SMOTE(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, smote.fit, X, Y)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    smote = SMOTE(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, smote.fit, X, Y)


def test_smote_wrong_kind():
    """Test either if an error is raised when the wrong kind of SMOTE is
    given."""

    kind = 'rnd'
    smote = SMOTE(kind=kind, random_state=RND_SEED)
    assert_raises(ValueError, smote.fit_sample, X, Y)


def test_smote_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    smote = SMOTE(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(RuntimeWarning, smote.fit, X, y_single_class)


def test_smote_fit():
    """Test the fitting method"""

    # Create the object
    smote = SMOTE(random_state=RND_SEED)
    # Fit the data
    smote.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(smote.min_c_, 0)
    assert_equal(smote.maj_c_, 1)
    assert_equal(smote.stats_c_[0], 500)
    assert_equal(smote.stats_c_[1], 4500)


def test_smote_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    smote = SMOTE(random_state=RND_SEED)
    assert_raises(RuntimeError, smote.sample, X, Y)


def test_sample_regular():
    """Test sample function with regular SMOTE."""

    # Create the object
    kind = 'regular'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'smote_reg_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'smote_reg_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    """Test sample function with regular SMOTE and a ratio of 0.5."""

    # Create the object
    ratio = 0.5
    kind = 'regular'
    smote = SMOTE(ratio=ratio, random_state=RND_SEED, kind=kind)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'smote_reg_x_05.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'smote_reg_y_05.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_sample_borderline1():
    """Test sample function with borderline 1 SMOTE."""

    # Create the object
    kind = 'borderline1'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'smote_bor_1_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'smote_bor_1_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_sample_borderline2():
    """Test sample function with borderline 2 SMOTE."""

    # Create the object
    kind = 'borderline2'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'smote_bor_2_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'smote_bor_2_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_sample_svm():
    """Test sample function with SVM SMOTE."""

    # Create the object
    kind = 'svm'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'smote_svm_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'smote_svm_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    sm = SMOTE(random_state=RND_SEED)
    sm.fit(X, Y)
    assert_raises(RuntimeError, sm.sample, np.random.random((100, 40)),
                  np.array([0] * 50 + [1] * 50))
