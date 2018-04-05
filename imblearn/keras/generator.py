"""Implement generators for ``keras`` which will balance the data."""

try:
    import keras
except ImportError:
    # Skip the tests for the examples
    import pytest
    keras = pytest.importorskip('keras')
    raise ImportError("To use the imblearn.keras module, you need to install "
                      "keras.")

from sklearn.base import clone
from sklearn.utils import safe_indexing
from sklearn.utils import check_random_state
from sklearn.utils.testing import set_random_state

from ..under_sampling import RandomUnderSampler


# FIXME: add docstring for random_state using Substitution
class BalancedBatchGenerator(keras.utils.Sequence):
    """Create balanced batches when training a keras model.

    Create a keras ``Sequence`` which is given to ``fit_generator``. The
    sampler defines the sampling strategy used to balance the dataset ahead of
    creating the batch. The sampler should have an attribute
    ``return_indices``.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Original imbalanced dataset.

    y : ndarray, shape (n_samples,) or (n_samples, n_classes)
        Associated targets.

    sampler : object or None, optional (default=None)
        A sampler instance which has an attribute ``return_indices``.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    {random_state}

    Attributes
    ----------
    sampler_ : object
        The sampler used to balance the dataset.

    indices_ : ndarray, shape (n_samples, n_features)
        The indices of the samples selected during sampling.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> from imblearn.datasets import make_imbalance
    >>> X, y = make_imbalance(iris.data, iris.target, {0: 30, 1: 50, 2: 40})
    >>> y = keras.utils.to_categorical(y, 3)
    >>> import keras
    >>> model = keras.models.Sequential()
    >>> model.add(keras.layers.Dense(y.shape[1], input_dim=X.shape[1],
    ...                              activation='softmax'))
    >>> model.compile(optimizer='sgd', loss='categorical_crossentropy',
    ...               metrics=['accuracy'])
    >>> from imblearn.keras import BalancedBatchGenerator
    >>> from imblearn.under_sampling import NearMiss
    >>> training_generator = BalancedBatchGenerator(
    ...     X, y, sampler=NearMiss(), batch_size=10, random_state=42)
    >>> callback_history = model.fit_generator(generator=training_generator,
    ...                                        epochs=10, verbose=0)


    """
    def __init__(self, X, y, sampler=None, batch_size=32, random_state=None):
        self.X = X
        self.y = y
        self.sampler = sampler
        self.batch_size = batch_size
        self.random_state = random_state
        self._sample()

    def _sample(self):
        random_state = check_random_state(self.random_state)
        if self.sampler is None:
            self.sampler_ = RandomUnderSampler(return_indices=True,
                                               random_state=random_state)
        else:
            if not hasattr(self.sampler, 'return_indices'):
                raise ValueError("'sampler' needs to return the indices of "
                                 "the samples selected. Provide a sampler "
                                 "which has an attribute 'return_indices'.")
            self.sampler_ = clone(self.sampler)
            self.sampler_.set_params(return_indices=True)
            set_random_state(self.sampler_, random_state)

        _, _, self.indices_ = self.sampler_.fit_sample(self.X, self.y)
        # shuffle the indices since the sampler are packing them by class
        random_state.shuffle(self.indices_)

    def __len__(self):
        return int(self.indices_.size // self.batch_size)

    def __getitem__(self, index):
        return (safe_indexing(self.X,
                              self.indices_[index * self.batch_size:
                                            (index + 1) * self.batch_size]),
                safe_indexing(self.y,
                              self.indices_[index * self.batch_size:
                                            (index + 1) * self.batch_size]))


# def balanced_batch_generator(X, y, sampler=None, batch_size=64,
#                              stratify=True):
#     """Create a balanced batch generator which can be plugged in
#     ``keras.fit_genertor``.

#     Parameters
#     ----------

#     """
#     if sampler is None:
#         sampler = RandomUnderSampler()
#     else:
#         if not hasattr(sampler, 'return_indices'):
#             raise ValueError("'sampler' needs to return the indices of "
#                              "the samples selected. Provide a sampler which "
#                              "has an attribute 'return_indices'.")
#         sampler.set_params(return_indices=True)

#     def generator(X=X, y=y, indices=indices, batch_size=batch_size,
#                   stratify=stratify):


#     _, _, indices = sampler.fit_sample(X, y)
