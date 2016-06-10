"""Test the module SMOTE."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from sklearn.datasets import make_classification

from unbalanced_dataset.over_sampling import SMOTE

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


def test_smote_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    assert_raises(ValueError, SMOTE, ratio=ratio)

    # Define a ratio greater than 1
    ratio = 100.0
    assert_raises(ValueError, SMOTE, ratio=ratio)

    # Define ratio as an unknown string
    ratio = 'rnd'
    assert_raises(ValueError, SMOTE, ratio=ratio)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    assert_raises(ValueError, SMOTE, ratio=ratio)


def test_smote_wrong_kind():
    """Test either if an error is raised when the wrong kind of SMOTE is
    given."""

    kind = 'rnd'
    assert_raises(ValueError, SMOTE, kind=kind)


def test_smote_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    smote = SMOTE(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_raises(RuntimeError, smote.fit, X, y_single_class)


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


def test_smote_transform_wt_fit():
    """Test either if an error is raised when transform is called before
    fitting"""

    # Create the object
    smote = SMOTE(random_state=RND_SEED)
    assert_raises(RuntimeError, smote.transform, X, Y)


def test_transform_regular():
    """Test transform function with regular SMOTE."""

    # Create the object
    kind = 'regular'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'smote_reg_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'smote_reg_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_transform_regular_half():
    """Test transform function with regular SMOTE and a ratio of 0.5."""

    # Create the object
    ratio = 0.5
    kind = 'regular'
    smote = SMOTE(ratio=ratio, random_state=RND_SEED, kind=kind)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'smote_reg_x_05.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'smote_reg_y_05.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_transform_borderline1():
    """Test transform function with borderline 1 SMOTE."""

    # Create the object
    kind = 'borderline1'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'smote_bor_1_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'smote_bor_1_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_transform_borderline2():
    """Test transform function with borderline 2 SMOTE."""

    # Create the object
    kind = 'borderline2'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'smote_bor_2_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'smote_bor_2_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_transform_svm():
    """Test transform function with SVM SMOTE."""

    # Create the object
    kind = 'svm'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'smote_svm_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'smote_svm_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
