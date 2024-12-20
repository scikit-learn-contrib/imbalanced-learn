"""Base class for sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin
from sklearn.preprocessing import label_binarize
from sklearn.utils._metadata_requests import METHODS
from sklearn.utils.multiclass import check_classification_targets

from .utils import check_sampling_strategy, check_target_type
from .utils._sklearn_compat import _fit_context, get_tags, validate_data
from .utils._validation import ArraysTransformer

if "fit_predict" not in METHODS:
    METHODS.append("fit_predict")
if "fit_transform" not in METHODS:
    METHODS.append("fit_transform")
METHODS.append("fit_resample")

try:
    from sklearn.utils._metadata_requests import SIMPLE_METHODS

    SIMPLE_METHODS.append("fit_resample")
except ImportError:
    # in older versions of scikit-learn, only METHODS is used
    pass


class SamplerMixin(metaclass=ABCMeta):
    """Mixin class for samplers with abstract method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = "sampler"

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, **params):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        X, y, _ = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_resample(self, X, y, **params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        output = self._fit_resample(X, y, **params)

        y_ = (
            label_binarize(output[1], classes=np.unique(y)) if binarize_y else output[1]
        )

        X_, y_ = arrays_transformer.transform(output[0], y_)
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    @abstractmethod
    def _fit_resample(self, X, y, **params):
        """Base method defined in each sampler to defined the sampling
        strategy.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray of shape (n_samples_new,)
            The corresponding label of `X_resampled`.

        """
        pass


class BaseSampler(SamplerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy

    def _check_X_y(self, X, y, accept_sparse=None):
        if accept_sparse is None:
            accept_sparse = ["csr", "csc"]
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = validate_data(self, X=X, y=y, reset=True, accept_sparse=accept_sparse)
        return X, y, binarize_y

    def fit(self, X, y, **params):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        return super().fit(X, y, **params)

    def fit_resample(self, X, y, **params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        return super().fit_resample(X, y, **params)

    def _more_tags(self):
        return {"X_types": ["2darray", "sparse", "dataframe"]}

    def __sklearn_tags__(self):
        from .utils._sklearn_compat import TargetTags
        from .utils._tags import InputTags, SamplerTags, Tags

        tags = Tags(
            estimator_type="sampler",
            target_tags=TargetTags(required=True),
            transformer_tags=None,
            regressor_tags=None,
            classifier_tags=None,
            sampler_tags=SamplerTags(),
        )
        tags.input_tags = InputTags()
        tags.input_tags.two_d_array = True
        tags.input_tags.sparse = True
        tags.input_tags.dataframe = True
        return tags


def _identity(X, y):
    return X, y


def is_sampler(estimator):
    """Return True if the given estimator is a sampler, False otherwise.

    Parameters
    ----------
    estimator : object
        Estimator to test.

    Returns
    -------
    is_sampler : bool
        True if estimator is a sampler, otherwise False.
    """

    if hasattr(estimator, "_estimator_type") and estimator._estimator_type == "sampler":
        return True
    tags = get_tags(estimator)
    if hasattr(tags, "sampler_tags") and tags.sampler_tags is not None:
        return True
    return False


class FunctionSampler(BaseSampler):
    """Construct a sampler from calling an arbitrary callable.

    Read more in the :ref:`User Guide <function_sampler>`.

    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. This will be passed the
        same arguments as transform, with args and kwargs forwarded. If func is
        None, then func will be the identity function.

    accept_sparse : bool, default=True
        Whether sparse input are supported. By default, sparse inputs are
        supported.

    kw_args : dict, default=None
        The keyword argument expected by ``func``.

    validate : bool, default=True
        Whether or not to bypass the validation of ``X`` and ``y``. Turning-off
        validation allows to use the ``FunctionSampler`` with any type of
        data.

        .. versionadded:: 0.6

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    sklearn.preprocessing.FunctionTransfomer : Stateless transformer.

    Notes
    -----
    See
    :ref:`sphx_glr_auto_examples_applications_plot_outlier_rejections.py`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from imblearn import FunctionSampler
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

    We can create to select only the first ten samples for instance.

    >>> def func(X, y):
    ...   return X[:10], y[:10]
    >>> sampler = FunctionSampler(func=func)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> np.all(X_res == X[:10])
    True
    >>> np.all(y_res == y[:10])
    True

    We can also create a specific function which take some arguments.

    >>> from collections import Counter
    >>> from imblearn.under_sampling import RandomUnderSampler
    >>> def func(X, y, sampling_strategy, random_state):
    ...   return RandomUnderSampler(
    ...       sampling_strategy=sampling_strategy,
    ...       random_state=random_state).fit_resample(X, y)
    >>> sampler = FunctionSampler(func=func,
    ...                           kw_args={'sampling_strategy': 'auto',
    ...                                    'random_state': 0})
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> print(f'Resampled dataset shape {sorted(Counter(y_res).items())}')
    Resampled dataset shape [(0, 100), (1, 100)]
    """

    _sampling_type = "bypass"

    _parameter_constraints: dict = {
        "func": [callable, None],
        "accept_sparse": ["boolean"],
        "kw_args": [dict, None],
        "validate": ["boolean"],
    }

    def __init__(self, *, func=None, accept_sparse=True, kw_args=None, validate=True):
        super().__init__()
        self.func = func
        self.accept_sparse = accept_sparse
        self.kw_args = kw_args
        self.validate = validate

    def fit(self, X, y):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        self._validate_params()
        # we need to overwrite SamplerMixin.fit to bypass the validation
        if self.validate:
            check_classification_targets(y)
            X, y, _ = self._check_X_y(X, y, accept_sparse=self.accept_sparse)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        return self

    def fit_resample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        self._validate_params()
        arrays_transformer = ArraysTransformer(X, y)

        if self.validate:
            check_classification_targets(y)
            X, y, binarize_y = self._check_X_y(X, y, accept_sparse=self.accept_sparse)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        output = self._fit_resample(X, y)

        if self.validate:
            y_ = (
                label_binarize(output[1], classes=np.unique(y))
                if binarize_y
                else output[1]
            )
            X_, y_ = arrays_transformer.transform(output[0], y_)
            return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

        return output

    def _fit_resample(self, X, y):
        func = _identity if self.func is None else self.func
        output = func(X, y, **(self.kw_args if self.kw_args else {}))
        return output
