"""Test the module CSS."""
# Authors: Bernhard Schlegel <bernhard.schlegel@mytum.de>
# License: MIT

from __future__ import print_function


import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_raises_regex,
                           assert_raises)

from imblearn.scaling import CSS

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[0.11622591, -0.0317206],
              [0.77481731, 0.60935141],
              [1.25192108, -0.22367336],
              [0.53366841, -0.30312976],
              [1.52091956, -0.49283504],
              [-0.28162401, -2.10400981],
              [0.83680821, 1.72827342],
              [0.3084254, 0.33299982],
              [0.70472253, -0.73309052],
              [0.28893132, -0.38761769],
              [1.15514042, 0.0129463],
              [0.88407872, 0.35454207],
              [1.31301027, -0.92648734],
              [-1.11515198, -0.93689695],
              [-0.18410027, -0.45194484],
              [0.9281014, 0.53085498],
              [-0.14374509, 0.27370049],
              [-0.41635887, -0.38299653],
              [0.08711622, 0.93259929],
              [1.70580611, -0.11219234]])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
R_TOL = 1e-4

def test_css_mode():
    # should two fail (illegal value for mode)
    css = CSS(mode='constant2', sampling_strategy='minority', c=1.01, shuffle=False)
    assert_raises(ValueError, css.fit_sample, X, y)

    css = CSS(mode='no mode', sampling_strategy='minority', c=0, shuffle=False)
    assert_raises(ValueError, css.fit_sample, X, y)

    # these two should not fail
    try:
        css = CSS(mode='constant', sampling_strategy='minority', c=0.25, shuffle=False)
        css.fit_sample(X,y)
        css = CSS(mode='linear', sampling_strategy='minority', c=0.25, shuffle=False)
        css.fit_sample(X,y)
    except Exception as e:
        raise ValueError('CSS raised an Exception unexpectedly! ({})'.format(e))

def test_css_target():
    # should two fail (illegal value for c)
    css = CSS(mode='constant', sampling_strategy='abc', c=0.5, shuffle=False)
    assert_raises(ValueError, css.fit_sample, X, y)

    # these three should not fail
    try:
        css = CSS(mode='constant', sampling_strategy='minority', c=0.05, shuffle=False)
        css.fit_sample(X,y)
        css = CSS(mode='constant', sampling_strategy='majority', c=0.05, shuffle=False)
        css.fit_sample(X,y)
        css = CSS(mode='constant', sampling_strategy='both', c=0.05, shuffle=False)
        css.fit_sample(X,y)
    except Exception as e:
        raise ValueError('CSS raised an Exception unexpectedly! ({})'.format(e))

def test_css_c():
    # should two fail (illegal value for c)
    css = CSS(mode='constant', sampling_strategy='minority', c=1.01, shuffle=False)
    assert_raises(ValueError, css.fit_sample, X, y)

    css = CSS(mode='constant', sampling_strategy='minority', c=0, shuffle=False)
    assert_raises(ValueError, css.fit_sample, X, y)

    # these two should not fail
    try:
        css = CSS(mode='constant', sampling_strategy='minority', c=0.01, shuffle=False)
        css.fit_sample(X,y)
        css = CSS(mode='linear', sampling_strategy='minority', c=0.99, shuffle=False)
        css.fit_sample(X,y)
    except Exception as e:
        raise ValueError('CSS raised an Exception unexpectedly! ({})'.format(e))


def test_sample_regular():
    # minority samples are unaffected when sampling_strategy is majority
    css = CSS(mode='constant', sampling_strategy='majority', c=1, shuffle=False)
    X_s, y_s = css.fit_sample(X,y)
    assert_allclose(X[y == 1], X_s[y_s == 1], rtol=R_TOL)

    # majority samples are unaffected when sampling_strategy is minority
    css = CSS(mode='constant', sampling_strategy='minority', c=1, shuffle=False)
    X_s, y_s = css.fit_sample(X,y)
    assert_allclose(X[y == 0], X_s[y_s == 0], rtol=R_TOL)

    # both are affected if sampling_strategy is both
    css = CSS(mode='constant', sampling_strategy='both', c=1, shuffle=False)
    X_s, y_s = css.fit_sample(X,y)
    if np.allclose(X[y == 0], X_s[y_s == 0], rtol=R_TOL):
        raise ValueError('np arrays should not be close!')

    # mathematical correctness of constant scaling majority (coarse)
    css = CSS(mode='constant', sampling_strategy='majority', c=1, shuffle=False)
    X_s, y_s = css.fit_sample(X, y)
    X_s_sub = X_s[y_s == 0]
    for i in range(2, len(X_s_sub)):
        if not abs(X_s_sub[0, 1] - X_s_sub[i, 1]) <= R_TOL:
            raise ValueError('numbers dont match')
        if not abs(X_s_sub[0, 0] - X_s_sub[i, 0]) <= R_TOL:
            raise ValueError('numbers dont match')

    # mathematical correctness of constant scaling majority (fine)
    c_test = 0.25
    css = CSS(mode='constant', sampling_strategy='majority', c=c_test, shuffle=False)
    X_s, y_s = css.fit_sample(X, y)
    X_sub = X[y==0]
    X_s_sub = X_s[y_s==0]
    mu = np.mean(X_s_sub, axis = 0)
    for i in range(0, len(X_s_sub)):
        if not abs(X_s_sub[i, 0] - (X_sub[i, 0] * (1 - c_test) + mu[0] * c_test)) <= R_TOL:
            raise ValueError('numbers dont match')
        if not abs(X_s_sub[i, 1] - (X_sub[i, 1] * (1 - c_test) + mu[1] * c_test)) <= R_TOL:
            raise ValueError('numbers dont match')
    # minority class should remain unaffected
    X_sub = X[y == 1]
    X_s_sub = X_s[y_s == 1]
    assert_allclose(X_sub, X_s_sub, rtol=R_TOL)

    # mathematical correctness of constant scaling minority (fine)
    c_test = 0.25
    css = CSS(mode='constant', sampling_strategy='minority', c=c_test, shuffle=False)
    X_s, y_s = css.fit_sample(X, y)
    X_sub = X[y==1]
    X_s_sub = X_s[y_s==1]
    mu = np.mean(X_s_sub, axis = 0)
    for i in range(0, len(X_s_sub)):
        if not abs(X_s_sub[i, 0] - (X_sub[i, 0] * (1 - c_test) + mu[0] * c_test)) <= R_TOL:
            raise ValueError('numbers dont match')
        if not abs(X_s_sub[i, 1] - (X_sub[i, 1] * (1 - c_test) + mu[1] * c_test)) <= R_TOL:
            raise ValueError('numbers dont match')
    # majority class should remain unaffected
    X_sub = X[y == 0]
    X_s_sub = X_s[y_s == 0]
    assert_allclose(X_sub, X_s_sub, rtol=R_TOL)

    # mathematical correctness of linear scaling both
    c_test = 0.1
    css = CSS(mode='linear', sampling_strategy='both', c=c_test, shuffle=False)
    X_s, y_s = css.fit_sample(X, y)
    for lvl in [0,1]:
        X_sub = X[y == lvl]
        X_s_sub = X_s[y_s == lvl]
        mu = np.mean(X_sub, axis=0)
        dists = abs(np.subtract(X_sub, mu))
        for i in range(0, len(X_s_sub)):
            for j in [0, 1]:
                norm = dists[i, j] * c_test + (1 - dists[i, j] * c_test)
                val_returned = X_s_sub[i, j]
                val_expected = X_sub[i, j] * (1 - dists[i, j] * c_test) / norm + mu[j] * dists[i, j] * c_test / norm
                if not abs(val_returned - val_expected) < R_TOL:
                    raise ValueError('numbers dont match')