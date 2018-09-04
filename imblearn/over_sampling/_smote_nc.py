"""Class to perform over-sampling using SMOTE-NC."""

# Authors: Dzianis Dudnik <ddudnik@protonmail.com>
# License: MIT
import warnings

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from . import SMOTE


class SMOTENC(SMOTE):
    """Class to perform over-sampling using SMOTE-NC.

    Implementation of the Synthetic Minority Over-sampling Technique
    for Nominal and Continuous (SMOTE-NC) features. SMOTE-NC is intended
    to deal with mixed datasets of categorical and numerical data.

    SMOTE-NC requires to one-hot encode the categorical features before
    sampling, i.e. using :class:`sklearn.preprocessing.OneHotEncoder`.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    {k_neighbors}

    {m_neighbors}

    {out_step}

    {kind}

    {svm_estimator}

    {n_jobs}

    {ratio}

    categorical_feature_indices : array-like, shape (n_categorical_features,)
        Indices to categorical feature ranges.
        Value of
        :attr:`sklearn.preprocessing.OneHotEncoder.feature_indices_`
        can be plugged directly.
        See :class:`sklearn.preprocessing.OneHotEncoder` for details.

    Attributes
    ----------
    std_median_ : float
        Median of standard deviations of continuous features.

    categorical_feature_indices_ : array-like, shape (n_categorical_features,)
        Indices to categorical feature ranges.

    continuous_feature_indices_ : array-like, shape (n_continuous_features,)
        Indices to continuous feature ranges.

    Notes
    -----
    See the original paper [1]_ for more details.

    Supports mutli-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See
    :ref:`sphx_glr_auto_examples_applications_plot_over_sampling_benchmark_lfw.py`,
    :ref:`sphx_glr_auto_examples_evaluation_plot_classification_report.py`,
    :ref:`sphx_glr_auto_examples_evaluation_plot_metrics.py`,
    :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`,
    :ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`,
    and :ref:`sphx_glr_auto_examples_over-sampling_plot_smote.py`.

    See also
    --------
    SMOTE : Over-sample using SMOTE.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    Examples
    --------

    >>> from collections import Counter
    >>> from numpy.random import RandomState
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> from imblearn.over_sampling import \
SMOTENC # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape (%s, %s)' % X.shape)
    Original dataset shape (1000, 20)
    >>> print('Original dataset samples per class {}'.format(Counter(y)))
    Original dataset samples per class Counter({1: 900, 0: 100})
    >>> # replace two last columns with categorical features encoded as integers
    >>> X[:, -2:] = RandomState(10).randint(0, 4, size=(1000, 2))
    >>> # One-hot encode the categorical columns
    >>> encoder = OneHotEncoder(n_values=[4, 4], categorical_features=[18, 19])
    >>> X = encoder.fit_transform(X)
    >>> print('One-hot encoded dataset shape (%s, %s)' % X.shape)
    One-hot encoded dataset shape (1000, 26)
    >>> sm = SMOTENC(random_state=42, categorical_feature_indices=encoder.feature_indices_)
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset samples per class {}'.format(Counter(y_res)))
    Resampled dataset samples per class Counter({0: 900, 1: 900})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 k_neighbors=5,
                 m_neighbors='deprecated',
                 out_step='deprecated',
                 kind='deprecated',
                 svm_estimator='deprecated',
                 n_jobs=1,
                 ratio=None,
                 categorical_feature_indices=None):
        super(SMOTENC, self).__init__(sampling_strategy=sampling_strategy,
                                      random_state=random_state,
                                      k_neighbors=k_neighbors,
                                      m_neighbors=m_neighbors,
                                      out_step=out_step,
                                      kind=kind,
                                      svm_estimator=svm_estimator,
                                      n_jobs=n_jobs,
                                      ratio=ratio)
        self.categorical_feature_indices = categorical_feature_indices

    def _fit_resample(self, X, y):
        if self.categorical_feature_indices is None:
            warnings.warn('No "categorical_feature_indices" were specified when '
                          'this instance was created. Will fall back '
                          'to normal SMOTE', RuntimeWarning)
            return super(SMOTENC, self)._fit_resample(X, y)

        feature_indices = check_array(self.categorical_feature_indices, ensure_2d=False,
                                      ensure_min_samples=2, estimator=self)
        n_features = X.shape[1]

        if np.any(feature_indices > n_features):
            raise ValueError('Indices of categorical features have to be less '
                             'than number of features in X: X.shape=(%s, %s)'
                             % X.shape)

        self.categorical_feature_indices_ = feature_indices
        self.continuous_feature_indices_ = np.setdiff1d(
            np.arange(n_features), np.arange(self.categorical_feature_indices_[0],
                                             self.categorical_feature_indices_[-1]))

        if self.continuous_feature_indices_.size == 0:
            raise ValueError('Looks like all features in X are '
                             'categorical which is not supported. '
                             'For this method to work X should have '
                             'at least 1 continuous feature.')

        if sparse.issparse(X):
            scaler = StandardScaler(with_mean=False,
                                    with_std=True,
                                    copy=False)
            scaler.fit(X.tocsc()[:, self.continuous_feature_indices_])
            self.std_median_ = np.median(np.sqrt(scaler.var_))
        else:
            std = np.std(X[:, self.continuous_feature_indices_], axis=0)
            self.std_median_ = np.median(std)

        return super(SMOTENC, self)._fit_resample(X, y)

    def _generate_sample(self, X, nn_data, nn_num, row, col, step):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        row : int
            Index pointing at feature vector in X which will be used
            as a base for creating new sample.

        col : int
            Index pointing at which nearest neighbor of base feature vector
            will be used when creating new sample.

        step : float
            Step size for new sample.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_features,)
            Single synthetically generated sample.

        """
        sample = super(SMOTENC, self)._generate_sample(X, nn_data, nn_num,
                                                       row, col, step)
        if not hasattr(self, "categorical_feature_indices_"):
            warnings.warn('No "categorical_feature_indices" were specified when '
                          'this instance was created. Will fall back '
                          'to normal SMOTE', RuntimeWarning)
            return sample

        is_sparse = sparse.issparse(nn_data)
        if is_sparse:
            nn_data = nn_data.tocsr()

        all_neighbors = nn_data[nn_num[row]]
        if is_sparse:
            all_neighbors = all_neighbors.tocsc()
            sample = sample.tolil()

        feature_idx_pairs = list(zip(self.categorical_feature_indices_[:-1],
                                     self.categorical_feature_indices_[1:]))
        for start, end in feature_idx_pairs:
            # FIXME: should sample for which neighbors were found be also
            # FIXME:  ... included when calculating most frequent nominal
            # FIXME:  ... feature values?
            nominal_values = all_neighbors[:, start:end]
            # FIXME: break ties randomly when several nominal values are
            # FIXME:  ... used by equal number of neighbors?
            most_used_value = np.argmax(np.sum(nominal_values, axis=0))
            new_nominal_value = np.zeros(nominal_values.shape[1])
            new_nominal_value[most_used_value] = 1
            if is_sparse:
                sample[:, start:end] = new_nominal_value
            else:
                sample[start:end] = new_nominal_value

        return sample.tocsr() if is_sparse else sample

    def _fit_nn_k(self, X):
        """Calls original method but on a modified copy if input."""
        return super(SMOTENC, self)._fit_nn_k(self._with_std_median(X))

    def _nn_k_neighbors(self, X):
        """Calls original method but on a modified copy if input."""
        return super(SMOTENC, self)._nn_k_neighbors(self._with_std_median(X))

    def _fit_nn_m(self, X):
        """Calls original method but on a modified copy if input."""
        return super(SMOTENC, self)._fit_nn_m(self._with_std_median(X))

    def _in_danger_noise(self, nn_estimator, samples, target_class, y,
                         kind='danger'):
        """Calls original method but on a modified copy if input."""
        return super(SMOTENC, self)._in_danger_noise(nn_estimator,
                                                     self._with_std_median(samples),
                                                     target_class, y, kind=kind)

    def _with_std_median(self, X):
        """
        Given that all categorical features are assumed to be one-hot encoded,
        their values are either 0 or 1. We replace values in original input
        which are equal to 1 with calculated median of standard deviations
        divided by 2. It will ensure that whenever distance is calculated
        between two feature vectors, the difference of two different categorical
        features will always equal to median standard deviation.
        """
        if not hasattr(self, "categorical_feature_indices_"):
            warnings.warn('No "categorical_feature_indices" were specified when '
                          'this instance was created. Will fallback '
                          'to normal SMOTE', RuntimeWarning)
            return X

        X_copy = X.copy().tolil() if sparse.issparse(X) else X.copy()
        start = self.categorical_feature_indices_[0]
        end = self.categorical_feature_indices_[-1]
        mask = X_copy[:, start:end] == 1
        X_copy[:, start:end][mask] = self.std_median_ / 2.0
        return X_copy.tocoo() if sparse.issparse(X_copy) else X_copy
