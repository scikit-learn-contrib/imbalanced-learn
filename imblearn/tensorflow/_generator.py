"""Implement generators for ``tensorflow`` which will balance the data."""

from __future__ import division

from sklearn.base import clone
from sklearn.utils import safe_indexing
from sklearn.utils import check_random_state
from sklearn.utils.testing import set_random_state

from ..under_sampling import RandomUnderSampler
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring


@Substitution(random_state=_random_state_docstring)
def balanced_batch_generator(X, y, sample_weight=None, sampler=None,
                             batch_size=32, random_state=None):
    """Create a balanced batch generator to train keras model.

    Returns a generator --- as well as the number of step per epoch --- which
    is given to ``fit_generator``. The sampler defines the sampling strategy
    used to balance the dataset ahead of creating the batch. The sampler should
    have an attribute ``return_indices``.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Original imbalanced dataset.

    y : ndarray, shape (n_samples,) or (n_samples, n_classes)
        Associated targets.

    sample_weight : ndarray, shape (n_samples,)
        Sample weight.

    sampler : object or None, optional (default=None)
        A sampler instance which has an attribute ``return_indices``.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    {random_state}

    Returns
    -------
    generator : generator of tuple
        Generate batch of data. The tuple generated are either (X_batch,
        y_batch) or (X_batch, y_batch, sampler_weight_batch).

    steps_per_epoch : int
        The number of samples per epoch. Required by ``fit_generator`` in
        keras.

    """
    random_state = check_random_state(random_state)
    if sampler is None:
        sampler_ = RandomUnderSampler(return_indices=True,
                                      random_state=random_state)
    else:
        if not hasattr(sampler, 'return_indices'):
            raise ValueError("'sampler' needs to return the indices of "
                             "the samples selected. Provide a sampler "
                             "which has an attribute 'return_indices'.")
        sampler_ = clone(sampler)
        sampler_.set_params(return_indices=True)
        set_random_state(sampler_, random_state)

    _, _, indices = sampler_.fit_sample(X, y)
    # shuffle the indices since the sampler are packing them by class
    random_state.shuffle(indices)

    def generator(X, y, sample_weight, indices, batch_size):
        if sample_weight is None:
            while True:
                for index in range(0, len(indices), batch_size):
                    yield (safe_indexing(X, indices[index:index + batch_size]),
                           safe_indexing(y, indices[index:index + batch_size]))
        else:
            while True:
                for index in range(0, len(indices), batch_size):
                    yield (safe_indexing(X, indices[index:index + batch_size]),
                           safe_indexing(y, indices[index:index + batch_size]),
                           safe_indexing(sample_weight,
                                         indices[index:index + batch_size]))

    return (generator(X, y, sample_weight, indices, batch_size),
            int(indices.size // batch_size))
