import pytest

keras = pytest.importorskip('keras')

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.datasets import load_iris

from imblearn.datasets import make_imbalance
from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss

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
    model = _build_keras_model(y.shape[1], X.shape[1])
    with pytest.raises(ValueError, match='needs to return the indices'):
        training_generator = BalancedBatchGenerator(X, y,
                                                    sampler=ClusterCentroids(),
                                                    batch_size=10,
                                                    random_state=42)
        model.fit_generator(generator=training_generator,
                            epochs=10)


@pytest.mark.parametrize(
    "sampler",
    [None, NearMiss()]
)
def test_balanced_batch_generator_class(sampler):
    model = _build_keras_model(y.shape[1], X.shape[1])
    training_generator = BalancedBatchGenerator(X, y,
                                                sampler=sampler,
                                                batch_size=10,
                                                random_state=42)
    model.fit_generator(generator=training_generator,
                        epochs=10)
