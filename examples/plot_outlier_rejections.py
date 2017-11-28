"""
===============================================================
Customized sampler to implement an outlier rejections estimator
===============================================================

This example illustrates the use of a custom sampler to implement an outlier
rejections estimator. It can be used easily within a pipeline in which the
number of samples can vary during training, which usually is a limitation of
the current scikit-learn pipeline.

"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from imblearn.misc import FunctionSampler
from imblearn.pipeline import make_pipeline

print(__doc__)

rng = np.random.RandomState(42)


def plot_scatter(X, y, title):
    plt.figure()
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class #1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class #0')
    plt.legend()
    plt.title(title)


# Generate contaminated training data
moons, _ = make_moons(n_samples=500, noise=0.05)
blobs, _ = make_blobs(n_samples=500, centers=[(-0.75, 2.25),
                                              (1.0, 2.0)],
                      cluster_std=0.25)
outliers = rng.uniform(low=-3, high=3, size=(500, 2))
X_train = np.vstack([moons, blobs, outliers])
y_train = np.hstack([np.ones(moons.shape[0], dtype=np.int8),
                     np.zeros(blobs.shape[0], dtype=np.int8),
                     rng.randint(0, 2, size=outliers.shape[0],
                                 dtype=np.int8)])

plot_scatter(X_train, y_train, 'Training dataset')

# Generate non-contaminated testing data
moons, _ = make_moons(n_samples=50, noise=0.05)
blobs, _ = make_blobs(n_samples=50, centers=[(-0.75, 2.25),
                                             (1.0, 2.0)],
                      cluster_std=0.25)
X_test = np.vstack([moons, blobs])
y_test = np.hstack([np.ones(moons.shape[0], dtype=np.int8),
                    np.zeros(blobs.shape[0], dtype=np.int8)])

plot_scatter(X_test, y_test, 'Testing dataset')


def outlier_rejection(X, y):
    model = IsolationForest(max_samples=100,
                            contamination=0.4,
                            random_state=rng)
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]


reject_sampler = FunctionSampler(func=outlier_rejection)
X_inliers, y_inliers = reject_sampler.fit_sample(X_train, y_train)
plot_scatter(X_inliers, y_inliers, 'Training data without outliers')

pipe = make_pipeline(FunctionSampler(func=outlier_rejection),
                     LogisticRegression(random_state=rng))
y_pred = pipe.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_pred))

clf = LogisticRegression(random_state=rng)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_pred))

plt.show()
