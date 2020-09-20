"""Class to perform over-sampling using ROSE."""

import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state
from .base import BaseOverSampler
from ..utils._validation import _deprecate_positional_args


class ROSE(BaseOverSampler):
    """Random Over-Sampling Examples (ROSE).

    This object is the implementation of ROSE algorithm.
    It generates new samples by a smoothed bootstrap approach,
    taking a random subsample of original data and adding a
    multivariate kernel density estimate :math:`f(x|Y_i)` around
    them with a smoothing matrix :math:`H_j`, and finally sampling
    from this distribution. A shrinking matrix can be provided, to
    set the bandwidth of the gaussian kernel.

    Read more in the :ref:`User Guide <rose>`.

    Parameters
    ----------
    sampling_strategy : float, str, dict or callable, default='auto'
        Sampling information to resample the data set.

        - When ``float``, it corresponds to the desired ratio of the number of
          samples in the minority class over the number of samples in the
          majority class after resampling. Therefore, the ratio is expressed as
          :math:`\\alpha_{os} = N_{rm} / N_{M}` where :math:`N_{rm}` is the
          number of samples in the minority class after resampling and
          :math:`N_{M}` is the number of samples in the majority class.

            .. warning::
               ``float`` is only available for **binary** classification. An
               error is raised for multi-class classification.

        - When ``str``, specify the class targeted by the resampling. The
          number of samples in the different classes will be equalized.
          Possible choices are:

            ``'minority'``: resample only the minority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not majority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.

    shrink_factors : dict, default= 1 for every class
        Dict of {classes: shrinkfactors} items, applied to
        the gaussian kernels. It can be used to compress/dilate the kernel.

    random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    See Also
    --------
    SMOTE : Over-sample using SMOTE.

    Notes
    -----

    References
    ----------
    .. [1] N. Lunardon, G. Menardi, N.Torelli, "ROSE: A Package for Binary
       Imbalanced Learning," R Journal, 6(1), 2014.

    .. [2] G Menardi, N. Torelli, "Training and assessing classification
       rules with imbalanced data," Data Mining and Knowledge
       Discovery, 28(1), pp.92-122, 2014.

    Examples
    --------

    >>> from imblearn.over_sampling import ROSE
    >>> from sklearn.datasets import make_classification
    >>> from collections import Counter
    >>> r = ROSE(shrink_factors={0:1, 1:0.5, 2:0.7})
    >>> X, y = make_classification(n_classes=3, class_sep=2,
    ... weights=[0.1, 0.7, 0.2], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=2000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({1: 1400, 2: 400, 0: 200})
    >>> X_res, y_res = r.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({2: 1400, 1: 1400, 0: 1400})
    """

    @_deprecate_positional_args
    def __init__(self, *, sampling_strategy="auto", shrink_factors=None,
                 random_state=None, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.shrink_factors = shrink_factors
        self.n_jobs = n_jobs

    def _make_samples(self,
                      X,
                      class_indices,
                      n_class_samples,
                      h_shrink):
        """ A support function that returns artificial samples constructed
        from a random subsample of the data, by adding a multiviariate
        gaussian kernel and sampling from this distribution. An optional
        shrink factor can be included, to compress/dilate the kernel.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Observations from which the samples will be created.

        class_indices : ndarray, shape (n_class_samples,)
            The target class indices

        n_class_samples : int
            The total number of samples per class to generate

        h_shrink : int
            the shrink factor

        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_samples, n_features)
            Synthetically generated samples.

        y_new : ndarray, shape (n_samples,)
            Target values for synthetic samples.

        """

        number_of_features = X.shape[1]
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.choice(
            class_indices, size=n_class_samples, replace=True)
        minimize_amise = (4 / ((number_of_features + 2) * len(
            class_indices))) ** (1 / (number_of_features + 4))
        if sparse.issparse(X):
            variances = np.diagflat(
                np.std(X[class_indices, :].toarray(), axis=0, ddof=1))
        else:
            variances = np.diagflat(
                np.std(X[class_indices, :], axis=0, ddof=1))
        h_opt = h_shrink * minimize_amise * variances
        randoms = random_state.standard_normal(size=(n_class_samples,
                                                     number_of_features))
        Xrose = np.matmul(randoms, h_opt) + X[samples_indices, :]
        if sparse.issparse(X):
            return sparse.csr_matrix(Xrose)
        return Xrose

    def _fit_resample(self, X, y):

        X_resampled = X.copy()
        y_resampled = y.copy()

        if self.shrink_factors is None:
            self.shrink_factors = {
                key: 1 for key in self.sampling_strategy_.keys()}

        for class_sample, n_samples in self.sampling_strategy_.items():
            class_indices = np.flatnonzero(y == class_sample)
            n_class_samples = n_samples
            X_new = self._make_samples(X,
                                       class_indices,
                                       n_samples,
                                       self.shrink_factors[class_sample])
            y_new = np.array([class_sample] * n_class_samples)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.concatenate((X_resampled, X_new))

            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled.astype(X.dtype), y_resampled.astype(y.dtype)
