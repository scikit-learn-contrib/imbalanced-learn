"""
The :mod:`imblearn.pipeline` module implements utilities to build a
composite estimator, as a chain of transforms, samples and estimators.
"""
# Adapted from scikit-learn

# Author: Edouard Duchesnay
#         Gael Varoquaux
#         Virgile Fritsch
#         Alexandre Gramfort
#         Lars Buitinck
#         Christos Aridas
#         Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: BSD

from __future__ import division

from sklearn import pipeline
from sklearn.base import clone
from sklearn.externals import six
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_memory
from sklearn.pipeline import _name_estimators

__all__ = ['Pipeline', 'make_pipeline']


class Pipeline(pipeline.Pipeline):
    """Pipeline of transforms and resamples with a final estimator.

    Sequentially apply a list of transforms, sampling, and a final estimator.
    Intermediate steps of the pipeline must be transformers or resamplers,
    that is, they must implement fit, transform and sample methods.
    The samplers are only applied during fit.
    The final estimator only needs to implement fit.
    The transformers and samplers in the pipeline can be cached using
    ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing
        fit/transform/fit_sample) that are chained, in the order in which they
        are chained, with the last object an estimator.

    memory : Instance of joblib.Memory or string, optional (default=None)
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.


    Attributes
    ----------
    named_steps : dict
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py`

    See also
    --------
    make_pipeline : helper function to make pipeline.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split as tts
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.neighbors import KNeighborsClassifier as KNN
    >>> from sklearn.metrics import classification_report
    >>> from imblearn.over_sampling import SMOTE
    >>> from imblearn.pipeline import Pipeline # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> pca = PCA()
    >>> smt = SMOTE(random_state=42)
    >>> knn = KNN()
    >>> pipeline = Pipeline([('smt', smt), ('pca', pca), ('knn', knn)])
    >>> X_train, X_test, y_train, y_test = tts(X, y, random_state=42)
    >>> pipeline.fit(X_train, y_train) # doctest: +ELLIPSIS
    Pipeline(...)
    >>> y_hat = pipeline.predict(X_test)
    >>> print(classification_report(y_test, y_hat))
                 precision    recall  f1-score   support
    <BLANKLINE>
              0       0.87      1.00      0.93        26
              1       1.00      0.98      0.99       224
    <BLANKLINE>
    avg / total       0.99      0.98      0.98       250
    <BLANKLINE>

    """

    # BaseEstimator interface

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform") or
                     hasattr(t, "fit_sample")) or not
                    (hasattr(t, "transform") or hasattr(t, "sample"))):
                raise TypeError(
                    "All intermediate steps of the chain should "
                    "be estimators that implement fit and transform or sample "
                    "(but not both) '%s' (type %s) doesn't)" % (t, type(t)))

            if ((hasattr(t, "fit_sample") and hasattr(t, "fit_transform")) or
                    (hasattr(t, "sample") and hasattr(t, "transform"))):
                raise TypeError(
                    "All intermediate steps of the chain should "
                    "be estimators that implement fit and transform or sample."
                    " '%s' implements both)" % (t))

            if isinstance(t, pipeline.Pipeline):
                raise TypeError(
                    "All intermediate steps of the chain should not be"
                    " Pipelines")

        # We allow last estimator to be None as an identity transformation
        if estimator is not None and not hasattr(estimator, "fit"):
            raise TypeError("Last step of Pipeline should implement fit. "
                            "'%s' (type %s) doesn't" % (estimator,
                                                        type(estimator)))

    # Estimator interface

    def _fit(self, X, y=None, **fit_params):
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        fit_sample_one_cached = memory.cache(_fit_sample_one)

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        yt = y
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if transformer is None:
                pass
            else:
                if hasattr(memory, 'cachedir') and memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve
                    # backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transformer
                if (hasattr(cloned_transformer, "transform") or
                        hasattr(cloned_transformer, "fit_transform")):
                    Xt, fitted_transformer = fit_transform_one_cached(
                        cloned_transformer, None, Xt, yt,
                        **fit_params_steps[name])
                elif hasattr(cloned_transformer, "sample"):
                    Xt, yt, fitted_transformer = fit_sample_one_cached(
                        cloned_transformer, Xt, yt, **fit_params_steps[name])
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator is None:
            return Xt, yt, {}
        return Xt, yt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms/samplers one after the other and
        transform/sample the data, then fit the transformed/sampled
        data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator

        """
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, yt, **fit_params)
        return self

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_sample(self, X, y=None, **fit_params):
        """Fit the model and sample with the final estimator

        Fits all the transformers/samplers one after the other and
        transform/sample the data, then uses fit_sample on transformed
        data with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples

        yt : array-like, shape = [n_samples, n_transformed_features]
            Transformed target

        """
        last_step = self._final_estimator
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        if last_step is None:
            return Xt
        elif hasattr(last_step, 'fit_sample'):
            return last_step.fit_sample(Xt, yt, **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def sample(self, X, y):
        """Sample the data with the final estimator

        Applies transformers/samplers to the data, and the sample
        method of the final estimator. Valid only if the final
        estimator implements sample.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None:
                continue
            if hasattr(transform, "fit_sample"):
                # XXX: Calling sample in pipeline it means that the
                # last estimator is a sampler. Samplers don't carry
                # the sampled data. So, call 'fit_sample' in all intermediate
                # steps to get the sampled data for the last estimator.
                Xt, y = transform.fit_sample(Xt, y)
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].fit_sample(Xt, y)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X):
        """Apply transformers/samplers to the data, and predict with the final
        estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_pred : array-like

        """
        Xt = X
        for _, transform in self.steps[:-1]:
            if transform is None:
                continue
            if hasattr(transform, "fit_sample"):
                pass
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        """Apply transformers/samplers, and predict_proba of the final
        estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]

        """
        Xt = X
        for _, transform in self.steps[:-1]:
            if transform is None:
                continue
            if hasattr(transform, "fit_sample"):
                pass
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):
        """Apply transformers/samplers, and decision_function of the final
        estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]

        """
        Xt = X
        for _, transform in self.steps[:-1]:
            if transform is None:
                continue
            if hasattr(transform, "fit_sample"):
                pass
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].decision_function(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X):
        """Apply transformers/samplers, and predict_log_proba of the final
        estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]

        """
        Xt = X
        for _, transform in self.steps[:-1]:
            if transform is None:
                continue
            if hasattr(transform, "fit_sample"):
                pass
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_log_proba(Xt)

    def _transform(self, X):
        Xt = X
        for name, transform in self.steps:
            if transform is None:
                continue
            if hasattr(transform, "fit_sample"):
                pass
            else:
                Xt = transform.transform(Xt)
        return Xt

    def _inverse_transform(self, X):
        Xt = X
        for name, transform in self.steps[::-1]:
            if transform is None:
                continue
            if hasattr(transform, "fit_sample"):
                pass
            else:
                Xt = transform.inverse_transform(Xt)
        return Xt

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, sample_weight=None):
        """Apply transformers/samplers, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
        """
        Xt = X
        for _, transform in self.steps[:-1]:
            if transform is None:
                continue
            if hasattr(transform, "fit_sample"):
                pass
            else:
                Xt = transform.transform(Xt)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, y, **score_params)


def _fit_transform_one(transformer, weight, X, y, **fit_params):
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res, transformer
    return res * weight, transformer


def _fit_sample_one(sampler, X, y, **fit_params):
    X_res, y_res = sampler.fit_sample(X, y, **fit_params)

    return X_res, y_res, sampler


def make_pipeline(*steps, **kwargs):
    """Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators,

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    ...     # doctest: +NORMALIZE_WHITESPACE
    Pipeline(memory=None,
             steps=[('standardscaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('gaussiannb', GaussianNB(priors=None))])

    Returns
    -------
    p : Pipeline
    """
    memory = kwargs.pop('memory', None)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return Pipeline(_name_estimators(steps), memory=memory)
