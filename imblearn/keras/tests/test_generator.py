import pytest

import numpy as np
from scipy import sparse

from sklearn.datasets import load_iris

keras = pytest.importorskip('keras')
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss

from imblearn.keras import BalancedBatchGenerator
from imblearn.keras import balanced_batch_generator

iris = load_iris()
X, y = make_imbalance(iris.data, iris.target, {0: 30, 1: 50, 2: 40})
y = to_categorical(y, 3)


def _build_keras_model(n_classes, n_features):
    model = Sequential()
    model.add(Dense(n_classes, input_dim=n_features, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def test_balanced_batch_generator_class_no_return_indices():
    with pytest.raises(ValueError, match='needs to return the indices'):
        BalancedBatchGenerator(X, y, sampler=ClusterCentroids(), batch_size=10)


@pytest.mark.parametrize(
    "sampler, sample_weight",
    [(None, None),
     (NearMiss(), None),
     (None, np.random.uniform(size=(y.shape[0])))]
)
def test_balanced_batch_generator_class(sampler, sample_weight):
    model = _build_keras_model(y.shape[1], X.shape[1])
    training_generator = BalancedBatchGenerator(X, y,
                                                sample_weight=sample_weight,
                                                sampler=sampler,
                                                batch_size=10,
                                                random_state=42)
    model.fit_generator(generator=training_generator,
                        epochs=10)


@pytest.mark.parametrize("is_sparse", [True, False])
def test_balanced_batch_generator_class_sparse(is_sparse):
    training_generator = BalancedBatchGenerator(sparse.csr_matrix(X), y,
                                                batch_size=10,
                                                sparse=is_sparse,
                                                random_state=42)
    for idx in range(len(training_generator)):
        X_batch, y_batch = training_generator.__getitem__(idx)
        if is_sparse:
            assert sparse.issparse(X_batch)
        else:
            assert not sparse.issparse(X_batch)


def test_balanced_batch_generator_function_no_return_indices():
    with pytest.raises(ValueError, match='needs to return the indices'):
        balanced_batch_generator(
            X, y, sampler=ClusterCentroids(), batch_size=10, random_state=42)


@pytest.mark.parametrize(
    "sampler, sample_weight",
    [(None, None),
     (NearMiss(), None),
     (None, np.random.uniform(size=(y.shape[0])))]
)
def test_balanced_batch_generator_function(sampler, sample_weight):
    model = _build_keras_model(y.shape[1], X.shape[1])
    training_generator, steps_per_epoch = balanced_batch_generator(
        X, y, sample_weight=sample_weight, sampler=sampler, batch_size=10,
        random_state=42)
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=10)


@pytest.mark.parametrize("is_sparse", [True, False])
def test_balanced_batch_generator_function_sparse(is_sparse):
    training_generator, steps_per_epoch = balanced_batch_generator(
        sparse.csr_matrix(X), y, sparse=is_sparse, batch_size=10,
        random_state=42)
    for idx in range(steps_per_epoch):
        X_batch, y_batch = next(training_generator)
        if is_sparse:
            assert sparse.issparse(X_batch)
        else:
            assert not sparse.issparse(X_batch)
