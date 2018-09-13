"""Implement generators for ``tensorflow`` which will balance the data."""

from __future__ import division

from scipy.sparse import issparse

from sklearn.base import clone
from sklearn.utils import safe_indexing
from sklearn.utils import check_random_state
from sklearn.utils.testing import set_random_state

from ..under_sampling import RandomUnderSampler
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring

DONT_HAVE_RANDOM_STATE = ('NearMiss', 'EditedNearestNeighbours',
                          'RepeatedEditedNearestNeighbours', 'AllKNN',
                          'NeighbourhoodCleaningRule', 'TomekLinks')


@Substitution(random_state=_random_state_docstring)
def balanced_batch_generator(X, y, sample_weight=None, sampler=None,
                             batch_size=32, keep_sparse=False,
                             random_state=None):
    """Create a balanced batch generator to train keras model.

    Returns a generator --- as well as the number of step per epoch --- which
    is given to ``fit_generator``. The sampler defines the sampling strategy
    used to balance the dataset ahead of creating the batch. The sampler should
    have an attribute ``sample_indices_``.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Original imbalanced dataset.

    y : ndarray, shape (n_samples,) or (n_samples, n_classes)
        Associated targets.

    sample_weight : ndarray, shape (n_samples,)
        Sample weight.

    sampler : object or None, optional (default=RandomUnderSampler)
        A sampler instance which has an attribute ``sample_indices_``.
        By default, the sampler used is a
        :class:`imblearn.under_sampling.RandomUnderSampler`.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    keep_sparse : bool, optional (default=False)
        Either or not to conserve or not the sparsity of the input ``X``. By
        default, the returned batches will be dense.

    {random_state}

    Returns
    -------
    generator : generator of tuple
        Generate batch of data. The tuple generated are either (X_batch,
        y_batch) or (X_batch, y_batch, sampler_weight_batch).

    steps_per_epoch : int
        The number of samples per epoch.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> class_dict = dict()
    >>> class_dict[0] = 30; class_dict[1] = 50; class_dict[2] = 40
    >>> from imblearn.datasets import make_imbalance
    >>> X, y = make_imbalance(X, y, class_dict)
    >>> X = X.astype(np.float32)
    >>> batch_size, learning_rate, epochs = 10, 0.01, 10
    >>> training_generator, steps_per_epoch = balanced_batch_generator(
    ...     X, y, sample_weight=None, sampler=None,
    ...     batch_size=batch_size, random_state=42)
    >>> input_size, output_size = X.shape[1], 3
    >>> import tensorflow as tf
    >>> def init_weights(shape):
    ...     return tf.Variable(tf.random_normal(shape, stddev=0.01))
    >>> def accuracy(y_true, y_pred):
    ...     return np.mean(np.argmax(y_pred, axis=1) == y_true)
    >>> # input and output
    >>> data = tf.placeholder("float32", shape=[None, input_size])
    >>> targets = tf.placeholder("int32", shape=[None])
    >>> # build the model and weights
    >>> W = init_weights([input_size, output_size])
    >>> b = init_weights([output_size])
    >>> out_act = tf.nn.sigmoid(tf.matmul(data, W) + b)
    >>> # build the loss, predict, and train operator
    >>> cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    ...     logits=out_act, labels=targets)
    >>> loss = tf.reduce_sum(cross_entropy)
    >>> optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    >>> train_op = optimizer.minimize(loss)
    >>> predict = tf.nn.softmax(out_act)
    >>> # Initialization of all variables in the graph
    >>> init = tf.global_variables_initializer()
    >>> with tf.Session() as sess:
    ...     print('Starting training')
    ...     sess.run(init)
    ...     for e in range(epochs):
    ...         for i in range(steps_per_epoch):
    ...             X_batch, y_batch = next(training_generator)
    ...             feed_dict = dict()
    ...             feed_dict[data] = X_batch; feed_dict[targets] = y_batch
    ...             sess.run([train_op, loss], feed_dict=feed_dict)
    ...         # For each epoch, run accuracy on train and test
    ...         feed_dict = dict()
    ...         feed_dict[data] = X
    ...         predicts_train = sess.run(predict, feed_dict=feed_dict)
    ...         print("epoch: {{}} train accuracy: {{:.3f}}"
    ...               .format(e, accuracy(y, predicts_train)))
    ... # doctest: +ELLIPSIS
    Starting training
    [...

    """

    random_state = check_random_state(random_state)
    if sampler is None:
        sampler_ = RandomUnderSampler(random_state=random_state)
    else:
        sampler_ = clone(sampler)
        # FIXME: Remove in 0.6
        if sampler_.__class__.__name__ not in DONT_HAVE_RANDOM_STATE:
            set_random_state(sampler_, random_state)
    sampler_.fit_resample(X, y)
    if not hasattr(sampler_, 'sample_indices_'):
        raise ValueError("'sampler' needs to have an attribute "
                         "'sample_indices_'.")
    indices = sampler_.sample_indices_
    # shuffle the indices since the sampler are packing them by class
    random_state.shuffle(indices)

    def generator(X, y, sample_weight, indices, batch_size):
        while True:
            for index in range(0, len(indices), batch_size):
                X_res = safe_indexing(X, indices[index:index + batch_size])
                y_res = safe_indexing(y, indices[index:index + batch_size])
                if issparse(X_res) and not keep_sparse:
                    X_res = X_res.toarray()
                if sample_weight is None:
                    yield X_res, y_res
                else:
                    sw_res = safe_indexing(sample_weight,
                                           indices[index:index + batch_size])
                    yield X_res, y_res, sw_res

    return (generator(X, y, sample_weight, indices, batch_size),
            int(indices.size // batch_size))
