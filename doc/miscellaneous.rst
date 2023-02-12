.. _miscellaneous:

======================
Miscellaneous samplers
======================

.. currentmodule:: imblearn

.. _function_sampler:

Custom samplers
---------------

A fully customized sampler, :class:`FunctionSampler`, is available in
imbalanced-learn such that you can fast prototype your own sampler by defining
a single function. Additional parameters can be added using the attribute
``kw_args`` which accepts a dictionary. The following example illustrates how
to retain the 10 first elements of the array ``X`` and ``y``::

  >>> import numpy as np
  >>> from imblearn import FunctionSampler
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
  ...                            n_redundant=0, n_repeated=0, n_classes=3,
  ...                            n_clusters_per_class=1,
  ...                            weights=[0.01, 0.05, 0.94],
  ...                            class_sep=0.8, random_state=0)
  >>> def func(X, y):
  ...   return X[:10], y[:10]
  >>> sampler = FunctionSampler(func=func)
  >>> X_res, y_res = sampler.fit_resample(X, y)
  >>> np.all(X_res == X[:10])
  True
  >>> np.all(y_res == y[:10])
  True

In addition, the parameter ``validate`` controls input checking. For instance,
turning ``validate=False`` allows to pass any type of target ``y`` and do some
sampling for regression targets::

  >>> from sklearn.datasets import make_regression
  >>> X_reg, y_reg = make_regression(n_samples=100, random_state=42)
  >>> rng = np.random.RandomState(42)
  >>> def dummy_sampler(X, y):
  ...     indices = rng.choice(np.arange(X.shape[0]), size=10)
  ...     return X[indices], y[indices]
  >>> sampler = FunctionSampler(func=dummy_sampler, validate=False)
  >>> X_res, y_res = sampler.fit_resample(X_reg, y_reg)
  >>> y_res
  array([  41.49112498, -142.78526195,   85.55095317,  141.43321419,
           75.46571114,  -67.49177372,  159.72700509, -169.80498923,
          211.95889757,  211.95889757])

We illustrated the use of such sampler to implement an outlier rejection
estimator which can be easily used within a
:class:`~imblearn.pipeline.Pipeline`:
:ref:`sphx_glr_auto_examples_applications_plot_outlier_rejections.py`

.. _generators:

Custom generators
-----------------

Imbalanced-learn provides specific generators for TensorFlow and Keras which
will generate balanced mini-batches.

.. _tensorflow_generator:

TensorFlow generator
~~~~~~~~~~~~~~~~~~~~

The :func:`~imblearn.tensorflow.balanced_batch_generator` allows to generate
balanced mini-batches using an imbalanced-learn sampler which returns indices.

Let's first generate some data::

  >>> n_features, n_classes = 10, 2
  >>> X, y = make_classification(
  ...     n_samples=10_000, n_features=n_features, n_informative=2,
  ...     n_redundant=0, n_repeated=0, n_classes=n_classes,
  ...     n_clusters_per_class=1, weights=[0.1, 0.9],
  ...     class_sep=0.8, random_state=0
  ... )
  >>> X = X.astype(np.float32)

Then, we can create the generator that will yield mini-batches that will be
balanced::

  >>> from imblearn.under_sampling import RandomUnderSampler
  >>> from imblearn.tensorflow import balanced_batch_generator
  >>> training_generator, steps_per_epoch = balanced_batch_generator(
  ...     X,
  ...     y,
  ...     sample_weight=None,
  ...     sampler=RandomUnderSampler(),
  ...     batch_size=32,
  ...     random_state=42,
  ... )

The ``generator`` and ``steps_per_epoch`` are used during the training of a
Tensorflow model. We will illustrate how to use this generator. First, we can
define a logistic regression model which will be optimized by a gradient
descent::

  >>> import tensorflow as tf
  >>> # initialize the weights and intercept
  >>> normal_initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
  >>> coef = tf.Variable(normal_initializer(
  ...     shape=[n_features, n_classes]), dtype="float32"
  ... )
  >>> intercept = tf.Variable(
  ...     normal_initializer(shape=[n_classes]), dtype="float32"
  ... )
  >>> # define the model
  >>> def logistic_regression(X):
  ...     return tf.nn.softmax(tf.matmul(X, coef) + intercept)
  >>> # define the loss function
  >>> def cross_entropy(y_true, y_pred):
  ...     y_true = tf.one_hot(y_true, depth=n_classes)
  ...     y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
  ...     return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))
  >>> # define our metric
  >>> def balanced_accuracy(y_true, y_pred):
  ...     cm = tf.math.confusion_matrix(tf.cast(y_true, tf.int64), tf.argmax(y_pred, 1))
  ...     per_class = np.diag(cm) / tf.math.reduce_sum(cm, axis=1)
  ...     return np.mean(per_class)
  >>> # define the optimizer
  >>> optimizer = tf.optimizers.SGD(learning_rate=0.01)
  >>> # define the optimization step
  >>> def run_optimization(X, y):
  ...     with tf.GradientTape() as g:
  ...         y_pred = logistic_regression(X)
  ...         loss = cross_entropy(y, y_pred)
  ...     gradients = g.gradient(loss, [coef, intercept])
  ...     optimizer.apply_gradients(zip(gradients, [coef, intercept]))

Once initialized, the model is trained by iterating on balanced mini-batches of
data and minimizing the loss previously defined::

  >>> epochs = 10
  >>> for e in range(epochs):
  ...     y_pred = logistic_regression(X)
  ...     loss = cross_entropy(y, y_pred)
  ...     bal_acc = balanced_accuracy(y, y_pred)
  ...     print(f"epoch: {e}, loss: {loss:.3f}, accuracy: {bal_acc}")
  ...     for i in range(steps_per_epoch):
  ...         X_batch, y_batch = next(training_generator)
  ...         run_optimization(X_batch, y_batch)
  epoch: 0, ...

.. _keras_generator:

Keras generator
~~~~~~~~~~~~~~~

Keras provides an higher level API in which a model can be defined and train by
calling ``fit_generator`` method to train the model. To illustrate, we will
define a logistic regression model::

  >>> from tensorflow import keras
  >>> y = keras.utils.to_categorical(y, 3)
  >>> model = keras.Sequential()
  >>> model.add(
  ...     keras.layers.Dense(
  ...         y.shape[1], input_dim=X.shape[1], activation='softmax'
  ...     )
  ... )
  >>> model.compile(
  ...     optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']
  ... )

:func:`~imblearn.keras.balanced_batch_generator` creates a balanced
mini-batches generator with the associated number of mini-batches which will be
generated::

  >>> from imblearn.keras import balanced_batch_generator
  >>> training_generator, steps_per_epoch = balanced_batch_generator(
  ...     X, y, sampler=RandomUnderSampler(), batch_size=10, random_state=42
  ... )

Then, ``fit`` can be called passing the generator and the step::

  >>> callback_history = model.fit(
  ...     training_generator,
  ...     steps_per_epoch=steps_per_epoch,
  ...     epochs=10,
  ...     verbose=1,
  ... )
  Epoch 1/10 ...

The second possibility is to use
:class:`~imblearn.keras.BalancedBatchGenerator`. Only an instance of this class
will be passed to ``fit``::

  >>> from imblearn.keras import BalancedBatchGenerator
  >>> training_generator = BalancedBatchGenerator(
  ...     X, y, sampler=RandomUnderSampler(), batch_size=10, random_state=42
  ... )
  >>> callback_history = model.fit(
  ...     training_generator,
  ...     steps_per_epoch=steps_per_epoch,
  ...     epochs=10,
  ...     verbose=1,
  ... )
  Epoch 1/10 ...

.. topic:: References

  * :ref:`sphx_glr_auto_examples_applications_porto_seguro_keras_under_sampling.py`
