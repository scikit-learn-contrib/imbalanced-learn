import pytest

tf = pytest.importorskip('tensforflow')

from sklearn.datasets import load_iris

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss

from imblearn.tensforflow import balanced_batch_generator

iris = load_iris()
X, y = make_imbalance(iris.data, iris.target, {0: 30, 1: 50, 2: 40})
