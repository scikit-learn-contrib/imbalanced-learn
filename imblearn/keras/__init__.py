"""The :mod:`imblearn.keras` provides utilities to deal with imbalanced dataset
in keras."""

from ._generator import BalancedBatchGenerator
from ..tensorflow._generator import balanced_batch_generator

balanced_batch_generator.__doc__ = \
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

    random_state : int, RandomState instance or None, optional (default=None)
        Control the randomization of the algorithm
        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    Returns
    -------
    generator : generator of tuple
        Generate batch of data. The tuple generated are either (X_batch,
        y_batch) or (X_batch, y_batch, sampler_weight_batch).

    steps_per_epoch : int
        The number of samples per epoch. Required by ``fit_generator`` in
        keras.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> from imblearn.datasets import make_imbalance
    >>> X, y = make_imbalance(iris.data, iris.target, {{0: 30, 1: 50, 2: 40}})
    >>> y = keras.utils.to_categorical(y, 3)
    >>> import keras
    >>> model = keras.models.Sequential()
    >>> model.add(keras.layers.Dense(y.shape[1], input_dim=X.shape[1],
    ...                              activation='softmax'))
    >>> model.compile(optimizer='sgd', loss='categorical_crossentropy',
    ...               metrics=['accuracy'])
    >>> from imblearn.keras import balanced_batch_generator
    >>> from imblearn.under_sampling import NearMiss
    >>> training_generator, steps_per_epoch = balanced_batch_generator(
    ...     X, y, sampler=NearMiss(), batch_size=10, random_state=42)
    >>> callback_history = model.fit_generator(generator=training_generator,
    ...                                        steps_per_epoch=steps_per_epoch,
    ...                                        epochs=10, verbose=0)

    """

__all__ = ['BalancedBatchGenerator',
           'balanced_batch_generator']
