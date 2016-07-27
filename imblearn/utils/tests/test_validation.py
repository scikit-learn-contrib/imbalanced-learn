"""Tests for input validation functions"""

from collections import Counter

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_warns
from numpy.testing import assert_equal

from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.utils import check_target_type

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


def test_check_target_type():
    """Test to check the target type function"""

    # Check that an error is raised when non estimator are passed
    assert_raises(TypeError, check_target_type, 'Something', np.ones((100, 1)))

    # Check that an error is raised when an estimator is passed but not a
    # sampler
    assert_raises(TypeError, check_target_type, AdaBoostClassifier(),
                  np.ones((100, 1)))

    # Binary sampler case

    # continuous case
    y = np.linspace(0, 1, 5000)
    sm = SMOTE(random_state=RND_SEED)
    assert_warns(UserWarning, sm.fit, X, y)

    # multiclass case
    y = np.array([0] * 2000 + [1] * 2000 + [2] * 1000)
    sm = SMOTE(random_state=RND_SEED)
    assert_warns(UserWarning, sm.fit, X, y)

    # Multiclass sampler case

    # continuous case
    y = np.linspace(0, 1, 5000)
    rus = RandomUnderSampler(random_state=RND_SEED)
    assert_warns(UserWarning, rus.fit, X, y)

    # Make y to be multiclass
    y = Y.copy()
    y[0:1000] = 2

    # Resample the data
    rus = RandomUnderSampler(random_state=RND_SEED)
    X_resampled, y_resampled = rus.fit_sample(X, y)

    # Check the size of y
    count_y_res = Counter(y_resampled)
    assert_equal(count_y_res[0], 400)
    assert_equal(count_y_res[1], 400)
    assert_equal(count_y_res[2], 400)
