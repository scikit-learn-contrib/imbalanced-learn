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
import warnings
from contextlib import contextmanager
from copy import deepcopy

from sklearn import pipeline
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import Bunch
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.fixes import parse_version
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _routing_enabled,
    get_routing_for_object,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, check_memory

from .base import METHODS
from .utils._sklearn_compat import (
    _fit_context,
    _print_elapsed_time,
    _raise_for_params,
    get_tags,
    process_routing,
    sklearn_version,
    validate_params,
)

__all__ = ["Pipeline", "make_pipeline"]


@contextmanager
def _raise_or_warn_if_not_fitted(estimator):
    """A context manager to make sure a NotFittedError is raised, if a sub-estimator
    raises the error.
    Otherwise, we raise a warning if the pipeline is not fitted, with the deprecation.
    TODO(0.15): remove this context manager and replace with check_is_fitted.
    """
    try:
        yield
    except NotFittedError as exc:
        raise NotFittedError("Pipeline is not fitted yet.") from exc

    # we only get here if the above didn't raise
    try:
        check_is_fitted(estimator)
    except NotFittedError:
        warnings.warn(
            (
                "This Pipeline instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using other methods such as transform, "
                "predict, etc. This will raise an error in 0.15 instead of the current "
                "warning."
            ),
            FutureWarning,
        )


def _cached_transform(
    sub_pipeline, *, cache, param_name, param_value, transform_params
):
    """Transform a parameter value using a sub-pipeline and cache the result.
    Parameters
    ----------
    sub_pipeline : Pipeline
        The sub-pipeline to be used for transformation.
    cache : dict
        The cache dictionary to store the transformed values.
    param_name : str
        The name of the parameter to be transformed.
    param_value : object
        The value of the parameter to be transformed.
    transform_params : dict
        The metadata to be used for transformation. This passed to the
        `transform` method of the sub-pipeline.
    Returns
    -------
    transformed_value : object
        The transformed value of the parameter.
    """
    if param_name not in cache:
        # If the parameter is a tuple, transform each element of the
        # tuple. This is needed to support the pattern present in
        # `lightgbm` and `xgboost` where users can pass multiple
        # validation sets.
        if isinstance(param_value, tuple):
            cache[param_name] = tuple(
                sub_pipeline.transform(element, **transform_params)
                for element in param_value
            )
        else:
            cache[param_name] = sub_pipeline.transform(param_value, **transform_params)

    return cache[param_name]


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
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    it to 'passthrough' or ``None``.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing
        fit/transform/fit_resample) that are chained, in the order in which
        they are chained, with the last object an estimator.

    transform_input : list of str, default=None
        The names of the :term:`metadata` parameters that should be transformed by the
        pipeline before passing it to the step consuming it.

        This enables transforming some input arguments to ``fit`` (other than ``X``)
        to be transformed by the steps of the pipeline up to the step which requires
        them. Requirement is defined via :ref:`metadata routing <metadata_routing>`.
        For instance, this can be used to pass a validation set through the pipeline.

        You can only set this if metadata routing is enabled, which you
        can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

        .. versionadded:: 1.6

    memory : Instance of joblib.Memory or str, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : :class:`~sklearn.utils.Bunch`
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during first step `fit` method.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    See Also
    --------
    make_pipeline : Helper function to make pipeline.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py`

    .. warning::
       A surprising behaviour of the `imbalanced-learn` pipeline is that it
       breaks the `scikit-learn` contract where one expects
       `estimmator.fit_transform(X, y)` to be equivalent to
       `estimator.fit(X, y).transform(X)`.

       The semantic of `fit_resample` is to be applied only during the fit
       stage. Therefore, resampling will happen when calling `fit_transform`
       while it will only happen on the `fit` stage when calling `fit` and
       `transform` separately. Practically, `fit_transform` will lead to a
       resampled dataset while `fit` and `transform` will not.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split as tts
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.neighbors import KNeighborsClassifier as KNN
    >>> from sklearn.metrics import classification_report
    >>> from imblearn.over_sampling import SMOTE
    >>> from imblearn.pipeline import Pipeline
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print(f'Original dataset shape {Counter(y)}')
    Original dataset shape Counter({1: 900, 0: 100})
    >>> pca = PCA()
    >>> smt = SMOTE(random_state=42)
    >>> knn = KNN()
    >>> pipeline = Pipeline([('smt', smt), ('pca', pca), ('knn', knn)])
    >>> X_train, X_test, y_train, y_test = tts(X, y, random_state=42)
    >>> pipeline.fit(X_train, y_train)
    Pipeline(...)
    >>> y_hat = pipeline.predict(X_test)
    >>> print(classification_report(y_test, y_hat))
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       0.87      1.00      0.93        26
               1       1.00      0.98      0.99       224
    <BLANKLINE>
        accuracy                           0.98       250
       macro avg       0.93      0.99      0.96       250
    weighted avg       0.99      0.98      0.98       250
    <BLANKLINE>
    """

    _parameter_constraints: dict = {
        "steps": "no_validation",  # validated in `_validate_steps`
        "transform_input": [list, None],
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["boolean"],
    }

    def __init__(self, steps, *, transform_input=None, memory=None, verbose=False):
        self.steps = steps
        self.transform_input = transform_input
        self.memory = memory
        self.verbose = verbose

    # BaseEstimator interface

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None or t == "passthrough":
                continue

            is_transfomer = hasattr(t, "fit") and hasattr(t, "transform")
            is_sampler = hasattr(t, "fit_resample")
            is_not_transfomer_or_sampler = not (is_transfomer or is_sampler)

            if is_not_transfomer_or_sampler:
                raise TypeError(
                    "All intermediate steps of the chain should "
                    "be estimators that implement fit and transform or "
                    "fit_resample (but not both) or be a string 'passthrough' "
                    "'%s' (type %s) doesn't)" % (t, type(t))
                )

            if is_transfomer and is_sampler:
                raise TypeError(
                    "All intermediate steps of the chain should "
                    "be estimators that implement fit and transform or "
                    "fit_resample."
                    " '%s' implements both)" % (t)
                )

            if isinstance(t, pipeline.Pipeline):
                raise TypeError(
                    "All intermediate steps of the chain should not be Pipelines"
                )

        # We allow last estimator to be None as an identity transformation
        if (
            estimator is not None
            and estimator != "passthrough"
            and not hasattr(estimator, "fit")
        ):
            raise TypeError(
                "Last step of Pipeline should implement fit or be "
                "the string 'passthrough'. '%s' (type %s) doesn't"
                % (estimator, type(estimator))
            )

    def _iter(self, with_final=True, filter_passthrough=True, filter_resample=True):
        """Generate (idx, (name, trans)) tuples from self.steps.

        When `filter_passthrough` is `True`, 'passthrough' and None
        transformers are filtered out. When `filter_resample` is `True`,
        estimator with a method `fit_resample` are filtered out.
        """
        it = super()._iter(with_final, filter_passthrough)
        if filter_resample:
            return filter(lambda x: not hasattr(x[-1], "fit_resample"), it)
        else:
            return it

    def _get_metadata_for_step(self, *, step_idx, step_params, all_params):
        """Get params (metadata) for step `name`.

        This transforms the metadata up to this step if required, which is
        indicated by the `transform_input` parameter.

        If a param in `step_params` is included in the `transform_input` list,
        it will be transformed.

        Parameters
        ----------
        step_idx : int
            Index of the step in the pipeline.

        step_params : dict
            Parameters specific to the step. These are routed parameters, e.g.
            `routed_params[name]`. If a parameter name here is included in the
            `pipeline.transform_input`, then it will be transformed. Note that
            these parameters are *after* routing, so the aliases are already
            resolved.

        all_params : dict
            All parameters passed by the user. Here this is used to call
            `transform` on the slice of the pipeline itself.

        Returns
        -------
        dict
            Parameters to be passed to the step. The ones which should be
            transformed are transformed.
        """
        if (
            self.transform_input is None
            or not all_params
            or not step_params
            or step_idx == 0
        ):
            # we only need to process step_params if transform_input is set
            # and metadata is given by the user.
            return step_params

        sub_pipeline = self[:step_idx]
        sub_metadata_routing = get_routing_for_object(sub_pipeline)
        # here we get the metadata required by sub_pipeline.transform
        transform_params = {
            key: value
            for key, value in all_params.items()
            if key
            in sub_metadata_routing.consumes(
                method="transform", params=all_params.keys()
            )
        }
        transformed_params = dict()  # this is to be returned
        transformed_cache = dict()  # used to transform each param once
        # `step_params` is the output of `process_routing`, so it has a dict for each
        # method (e.g. fit, transform, predict), which are the args to be passed to
        # those methods. We need to transform the parameters which are in the
        # `transform_input`, before returning these dicts.
        for method, method_params in step_params.items():
            transformed_params[method] = Bunch()
            for param_name, param_value in method_params.items():
                # An example of `(param_name, param_value)` is
                # `('sample_weight', array([0.5, 0.5, ...]))`
                if param_name in self.transform_input:
                    # This parameter now needs to be transformed by the sub_pipeline, to
                    # this step. We cache these computations to avoid repeating them.
                    transformed_params[method][param_name] = _cached_transform(
                        sub_pipeline,
                        cache=transformed_cache,
                        param_name=param_name,
                        param_value=param_value,
                        transform_params=transform_params,
                    )
                else:
                    transformed_params[method][param_name] = param_value
        return transformed_params

    # Estimator interface

    # def _fit(self, X, y=None, **fit_params_steps):
    def _fit(self, X, y=None, routed_params=None, raw_params=None):
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        fit_resample_one_cached = memory.cache(_fit_resample_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False, filter_resample=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)

            # Fit or load from cache the current transformer
            step_params = self._get_metadata_for_step(
                step_idx=step_idx,
                step_params=routed_params[name],
                all_params=raw_params,
            )
            if hasattr(cloned_transformer, "transform") or hasattr(
                cloned_transformer, "fit_transform"
            ):
                X, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    weight=None,
                    message_clsname="Pipeline",
                    message=self._log_message(step_idx),
                    params=step_params,
                )
            elif hasattr(cloned_transformer, "fit_resample"):
                X, y, fitted_transformer = fit_resample_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    message_clsname="Pipeline",
                    message=self._log_message(step_idx),
                    params=routed_params[name],
                )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X, y

    # The `fit_*` methods need to be overridden to support the samplers.
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **params):
        """Fit the model.

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

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True` is set via
                :func:`~sklearn.set_config`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        self : Pipeline
            This estimator.
        """
        if not _routing_enabled() and self.transform_input is not None:
            raise ValueError(
                "The `transform_input` parameter can only be set if metadata "
                "routing is enabled. You can enable metadata routing using "
                "`sklearn.set_config(enable_metadata_routing=True)`."
            )

        if sklearn_version < parse_version("1.4") and self.transform_input is not None:
            raise ValueError(
                "The `transform_input` parameter is not supported in scikit-learn "
                "versions prior to 1.4. Please upgrade to scikit-learn 1.4 or "
                "later."
            )

        routed_params = self._check_method_params(method="fit", props=params)
        Xt, yt = self._fit(X, y, routed_params, raw_params=params)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_params = self._get_metadata_for_step(
                    step_idx=len(self) - 1,
                    step_params=routed_params[self.steps[-1][0]],
                    all_params=params,
                )
                self._final_estimator.fit(Xt, yt, **last_step_params["fit"])
        return self

    def _can_fit_transform(self):
        return (
            self._final_estimator == "passthrough"
            or hasattr(self._final_estimator, "transform")
            or hasattr(self._final_estimator, "fit_transform")
        )

    @available_if(_can_fit_transform)
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X, y=None, **params):
        """Fit the model and transform with the final estimator.

        Fits all the transformers/samplers one after the other and
        transform/sample the data, then uses fit_transform on
        transformed data with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        routed_params = self._check_method_params(method="fit_transform", props=params)
        Xt, yt = self._fit(X, y, routed_params)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            last_step_params = self._get_metadata_for_step(
                step_idx=len(self) - 1,
                step_params=routed_params[self.steps[-1][0]],
                all_params=params,
            )
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(
                    Xt, yt, **last_step_params["fit_transform"]
                )
            else:
                return last_step.fit(Xt, y, **last_step_params["fit"]).transform(
                    Xt, **last_step_params["transform"]
                )

    @available_if(pipeline._final_estimator_has("predict"))
    def predict(self, X, **params):
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters to the ``predict`` called at the end of all
                transformations in the pipeline.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionadded:: 0.20

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True` is set via
                :func:`~sklearn.set_config`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

            Note that while this may be used to return uncertainties from some
            models with ``return_std`` or ``return_cov``, uncertainties that are
            generated by the transformations in the pipeline are not propagated
            to the final estimator.

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        # TODO(0.15): Remove the context manager and use check_is_fitted(self)
        with _raise_or_warn_if_not_fitted(self):
            Xt = X

            if not _routing_enabled():
                for _, name, transform in self._iter(with_final=False):
                    Xt = transform.transform(Xt)
                return self.steps[-1][1].predict(Xt, **params)

            # metadata routing enabled
            routed_params = process_routing(self, "predict", **params)
            for _, name, transform in self._iter(with_final=False):
                Xt = transform.transform(Xt, **routed_params[name].transform)
            return self.steps[-1][1].predict(
                Xt, **routed_params[self.steps[-1][0]].predict
            )

    def _can_fit_resample(self):
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "fit_resample"
        )

    @available_if(_can_fit_resample)
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_resample(self, X, y=None, **params):
        """Fit the model and sample with the final estimator.

        Fits all the transformers/samplers one after the other and
        transform/sample the data, then uses fit_resample on transformed
        data with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_transformed_features)
            Transformed samples.

        yt : array-like of shape (n_samples, n_transformed_features)
            Transformed target.
        """
        routed_params = self._check_method_params(method="fit_resample", props=params)
        Xt, yt = self._fit(X, y, routed_params)
        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            last_step_params = routed_params[self.steps[-1][0]]
            if hasattr(last_step, "fit_resample"):
                return last_step.fit_resample(
                    Xt, yt, **last_step_params["fit_resample"]
                )

    @available_if(pipeline._final_estimator_has("fit_predict"))
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_predict(self, X, y=None, **params):
        """Apply `fit_predict` of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters to the ``predict`` called at the end of all
                transformations in the pipeline.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionadded:: 0.20

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

            Note that while this may be used to return uncertainties from some
            models with ``return_std`` or ``return_cov``, uncertainties that are
            generated by the transformations in the pipeline are not propagated
            to the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted target.
        """
        routed_params = self._check_method_params(method="fit_predict", props=params)
        Xt, yt = self._fit(X, y, routed_params)

        params_last_step = routed_params[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][-1].fit_predict(
                Xt, yt, **params_last_step.get("fit_predict", {})
            )
        return y_pred

    # TODO: remove the following methods when the minimum scikit-learn >= 1.4
    # They do not depend on resampling but we need to redefine them for the
    # compatibility with the metadata routing framework.
    @available_if(pipeline._final_estimator_has("predict_proba"))
    def predict_proba(self, X, **params):
        """Transform the data, and apply `predict_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_proba` method. Only valid if the final estimator implements
        `predict_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters to the `predict_proba` called at the end of all
                transformations in the pipeline.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionadded:: 0.20

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the final estimator.
        """
        # TODO(0.15): Remove the context manager and use check_is_fitted(self)
        with _raise_or_warn_if_not_fitted(self):
            Xt = X

            if not _routing_enabled():
                for _, name, transform in self._iter(with_final=False):
                    Xt = transform.transform(Xt)
                return self.steps[-1][1].predict_proba(Xt, **params)

            # metadata routing enabled
            routed_params = process_routing(self, "predict_proba", **params)
            for _, name, transform in self._iter(with_final=False):
                Xt = transform.transform(Xt, **routed_params[name].transform)
            return self.steps[-1][1].predict_proba(
                Xt, **routed_params[self.steps[-1][0]].predict_proba
            )

    @available_if(pipeline._final_estimator_has("decision_function"))
    def decision_function(self, X, **params):
        """Transform the data, and apply `decision_function` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `decision_function` method. Only valid if the final estimator
        implements `decision_function`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of string -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        y_score : ndarray of shape (n_samples, n_classes)
            Result of calling `decision_function` on the final estimator.
        """
        # TODO(0.15): Remove the context manager and use check_is_fitted(self)
        with _raise_or_warn_if_not_fitted(self):
            _raise_for_params(params, self, "decision_function")

            # not branching here since params is only available if
            # enable_metadata_routing=True
            routed_params = process_routing(self, "decision_function", **params)

            Xt = X
            for _, name, transform in self._iter(with_final=False):
                Xt = transform.transform(
                    Xt, **routed_params.get(name, {}).get("transform", {})
                )
            return self.steps[-1][1].decision_function(
                Xt,
                **routed_params.get(self.steps[-1][0], {}).get("decision_function", {}),
            )

    @available_if(pipeline._final_estimator_has("score_samples"))
    def score_samples(self, X):
        """Transform the data, and apply `score_samples` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score_samples` method. Only valid if the final estimator implements
        `score_samples`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Result of calling `score_samples` on the final estimator.
        """
        # TODO(0.15): Remove the context manager and use check_is_fitted(self)
        with _raise_or_warn_if_not_fitted(self):
            Xt = X
            for _, _, transformer in self._iter(with_final=False):
                Xt = transformer.transform(Xt)
            return self.steps[-1][1].score_samples(Xt)

    @available_if(pipeline._final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **params):
        """Transform the data, and apply `predict_log_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_log_proba` method. Only valid if the final estimator
        implements `predict_log_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters to the `predict_log_proba` called at the end of all
                transformations in the pipeline.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionadded:: 0.20

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_log_proba` on the final estimator.
        """
        # TODO(0.15): Remove the context manager and use check_is_fitted(self)
        with _raise_or_warn_if_not_fitted(self):
            Xt = X

            if not _routing_enabled():
                for _, name, transform in self._iter(with_final=False):
                    Xt = transform.transform(Xt)
                return self.steps[-1][1].predict_log_proba(Xt, **params)

            # metadata routing enabled
            routed_params = process_routing(self, "predict_log_proba", **params)
            for _, name, transform in self._iter(with_final=False):
                Xt = transform.transform(Xt, **routed_params[name].transform)
            return self.steps[-1][1].predict_log_proba(
                Xt, **routed_params[self.steps[-1][0]].predict_log_proba
            )

    def _can_transform(self):
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
        )

    @available_if(_can_transform)
    def transform(self, X, **params):
        """Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        # TODO(0.15): Remove the context manager and use check_is_fitted(self)
        with _raise_or_warn_if_not_fitted(self):
            _raise_for_params(params, self, "transform")

            # not branching here since params is only available if
            # enable_metadata_routing=True
            routed_params = process_routing(self, "transform", **params)
            Xt = X
            for _, name, transform in self._iter():
                Xt = transform.transform(Xt, **routed_params[name].transform)
            return Xt

    def _can_inverse_transform(self):
        return all(hasattr(t, "inverse_transform") for _, _, t in self._iter())

    @available_if(_can_inverse_transform)
    def inverse_transform(self, Xt, **params):
        """Apply `inverse_transform` for each step in a reverse order.

        All estimators in the pipeline must support `inverse_transform`.

        Parameters
        ----------
        Xt : array-like of shape (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Inverse transformed data, that is, data in the original feature
            space.
        """
        # TODO(0.15): Remove the context manager and use check_is_fitted(self)
        with _raise_or_warn_if_not_fitted(self):
            _raise_for_params(params, self, "inverse_transform")

            # we don't have to branch here, since params is only non-empty if
            # enable_metadata_routing=True.
            routed_params = process_routing(self, "inverse_transform", **params)
            reverse_iter = reversed(list(self._iter()))
            for _, name, transform in reverse_iter:
                Xt = transform.inverse_transform(
                    Xt, **routed_params[name].inverse_transform
                )
            return Xt

    @available_if(pipeline._final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None, **params):
        """Transform the data, and apply `score` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score` method. Only valid if the final estimator implements `score`.

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

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        score : float
            Result of calling `score` on the final estimator.
        """
        # TODO(0.15): Remove the context manager and use check_is_fitted(self)
        with _raise_or_warn_if_not_fitted(self):
            Xt = X
            if not _routing_enabled():
                for _, name, transform in self._iter(with_final=False):
                    Xt = transform.transform(Xt)
                score_params = {}
                if sample_weight is not None:
                    score_params["sample_weight"] = sample_weight
                return self.steps[-1][1].score(Xt, y, **score_params)

            # metadata routing is enabled.
            routed_params = process_routing(
                self, "score", sample_weight=sample_weight, **params
            )

            Xt = X
            for _, name, transform in self._iter(with_final=False):
                Xt = transform.transform(Xt, **routed_params[name].transform)
            return self.steps[-1][1].score(
                Xt, y, **routed_params[self.steps[-1][0]].score
            )

    # TODO: once scikit-learn >= 1.4, the following function should be simplified by
    # calling `super().get_metadata_routing()`
    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__)

        # first we add all steps except the last one
        for _, name, trans in self._iter(
            with_final=False, filter_passthrough=True, filter_resample=False
        ):
            method_mapping = MethodMapping()
            # fit, fit_predict, and fit_transform call fit_transform if it
            # exists, or else fit and transform
            if hasattr(trans, "fit_transform"):
                (
                    method_mapping.add(caller="fit", callee="fit_transform")
                    .add(caller="fit_transform", callee="fit_transform")
                    .add(caller="fit_predict", callee="fit_transform")
                )
            else:
                (
                    method_mapping.add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="fit_transform", callee="fit")
                    .add(caller="fit_transform", callee="transform")
                    .add(caller="fit_predict", callee="fit")
                    .add(caller="fit_predict", callee="transform")
                )

            (
                # handling sampler if the fit_* stage
                method_mapping.add(caller="fit", callee="fit_resample")
                .add(caller="fit_transform", callee="fit_resample")
                .add(caller="fit_predict", callee="fit_resample")
            )
            (
                method_mapping.add(caller="predict", callee="transform")
                .add(caller="predict", callee="transform")
                .add(caller="predict_proba", callee="transform")
                .add(caller="decision_function", callee="transform")
                .add(caller="predict_log_proba", callee="transform")
                .add(caller="transform", callee="transform")
                .add(caller="inverse_transform", callee="inverse_transform")
                .add(caller="score", callee="transform")
                .add(caller="fit_resample", callee="transform")
            )

            router.add(method_mapping=method_mapping, **{name: trans})

        final_name, final_est = self.steps[-1]
        if final_est is None or final_est == "passthrough":
            return router

        # then we add the last step
        method_mapping = MethodMapping()
        if hasattr(final_est, "fit_transform"):
            method_mapping.add(caller="fit_transform", callee="fit_transform")
        else:
            (
                method_mapping.add(caller="fit", callee="fit").add(
                    caller="fit", callee="transform"
                )
            )
        (
            method_mapping.add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="fit_predict", callee="fit_predict")
            .add(caller="predict_proba", callee="predict_proba")
            .add(caller="decision_function", callee="decision_function")
            .add(caller="predict_log_proba", callee="predict_log_proba")
            .add(caller="transform", callee="transform")
            .add(caller="inverse_transform", callee="inverse_transform")
            .add(caller="score", callee="score")
            .add(caller="fit_resample", callee="fit_resample")
        )

        router.add(method_mapping=method_mapping, **{final_name: final_est})
        return router

    def _check_method_params(self, method, props, **kwargs):
        if _routing_enabled():
            routed_params = process_routing(self, method, **props, **kwargs)
            return routed_params
        else:
            fit_params_steps = Bunch(
                **{
                    name: Bunch(**{method: {} for method in METHODS})
                    for name, step in self.steps
                    if step is not None
                }
            )
            for pname, pval in props.items():
                if "__" not in pname:
                    raise ValueError(
                        "Pipeline.fit does not accept the {} parameter. "
                        "You can pass parameters to specific steps of your "
                        "pipeline using the stepname__parameter format, e.g. "
                        "`Pipeline.fit(X, y, logisticregression__sample_weight"
                        "=sample_weight)`.".format(pname)
                    )
                step, param = pname.split("__", 1)
                fit_params_steps[step]["fit"][param] = pval
                # without metadata routing, fit_transform and fit_predict
                # get all the same params and pass it to the last fit.
                fit_params_steps[step]["fit_transform"][param] = pval
                fit_params_steps[step]["fit_predict"][param] = pval
            return fit_params_steps

    def __sklearn_is_fitted__(self):
        """Indicate whether pipeline has been fit.

        This is done by checking whether the last non-`passthrough` step of the
        pipeline is fitted.

        An empty pipeline is considered fitted.
        """

        # First find the last step that is not 'passthrough'
        last_step = None
        for _, estimator in reversed(self.steps):
            if estimator != "passthrough":
                last_step = estimator
                break

        if last_step is None:
            # All steps are 'passthrough', so the pipeline is considered fitted
            return True

        try:
            # check if the last step of the pipeline is fitted
            # we only check the last step since if the last step is fit, it
            # means the previous steps should also be fit. This is faster than
            # checking if every step of the pipeline is fit.
            check_is_fitted(last_step)
            return True
        except NotFittedError:
            return False

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()

        if not self.steps:
            return tags

        try:
            if self.steps[0][1] is not None and self.steps[0][1] != "passthrough":
                tags.input_tags.pairwise = get_tags(
                    self.steps[0][1]
                ).input_tags.pairwise
        except (ValueError, AttributeError, TypeError):
            # This happens when the `steps` is not a list of (name, estimator)
            # tuples and `fit` is not called yet to validate the steps.
            pass

        try:
            if self.steps[-1][1] is not None and self.steps[-1][1] != "passthrough":
                last_step_tags = get_tags(self.steps[-1][1])
                tags.estimator_type = last_step_tags.estimator_type
                tags.target_tags.multi_output = last_step_tags.target_tags.multi_output
                tags.classifier_tags = deepcopy(last_step_tags.classifier_tags)
                tags.regressor_tags = deepcopy(last_step_tags.regressor_tags)
                tags.transformer_tags = deepcopy(last_step_tags.transformer_tags)
        except (ValueError, AttributeError, TypeError):
            # This happens when the `steps` is not a list of (name, estimator)
            # tuples and `fit` is not called yet to validate the steps.
            pass

        return tags


def _fit_resample_one(sampler, X, y, message_clsname="", message=None, params=None):
    with _print_elapsed_time(message_clsname, message):
        X_res, y_res = sampler.fit_resample(X, y, **params.get("fit_resample", {}))

        return X_res, y_res, sampler


def _transform_one(transformer, X, y, weight, params=None):
    """Call transform and apply weight to output.

    Parameters
    ----------
    transformer : estimator
        Estimator to be used for transformation.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data to be transformed.

    y : ndarray of shape (n_samples,)
        Ignored.

    weight : float
        Weight to be applied to the output of the transformation.

    params : dict
        Parameters to be passed to the transformer's ``transform`` method.

        This should be of the form ``process_routing()["step_name"]``.
    """
    res = transformer.transform(X, **params.transform)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


def _fit_transform_one(
    transformer, X, y, weight, message_clsname="", message=None, params=None
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.

    ``params`` needs to be of the form ``process_routing()["step_name"]``.
    """
    params = params or {}
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, "fit_transform"):
            res = transformer.fit_transform(X, y, **params.get("fit_transform", {}))
        else:
            res = transformer.fit(X, y, **params.get("fit", {})).transform(
                X, **params.get("transform", {})
            )

    if weight is None:
        return res, transformer
    return res * weight, transformer


@validate_params(
    {
        "memory": [None, str, HasMethods(["cache"])],
        "transform_input": [None, list],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def make_pipeline(*steps, memory=None, transform_input=None, verbose=False):
    """Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators
        A list of estimators.

    memory : None, str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    transform_input : list of str, default=None
        This enables transforming some input arguments to ``fit`` (other than ``X``)
        to be transformed by the steps of the pipeline up to the step which requires
        them. Requirement is defined via :ref:`metadata routing <metadata_routing>`.
        This can be used to pass a validation set through the pipeline for instance.

        You can only set this if metadata routing is enabled, which you
        can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

        .. versionadded:: 1.6

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Returns
    -------
    p : Pipeline
        Returns an imbalanced-learn `Pipeline` instance that handles samplers.

    See Also
    --------
    imblearn.pipeline.Pipeline : Class for creating a pipeline of
        transforms with a final estimator.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('gaussiannb', GaussianNB())])
    """
    return Pipeline(
        pipeline._name_estimators(steps),
        memory=memory,
        transform_input=transform_input,
        verbose=verbose,
    )
