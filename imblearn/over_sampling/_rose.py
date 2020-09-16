"""Class to perform over-sampling using ROSE."""

import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state
from .base import BaseOverSampler
from ..utils._validation import _deprecate_positional_args
# from sklearn.utils import check_X_y


class ROSE(BaseOverSampler):

    """Oversample using Random OverSampling Examples (ROSE) algorithm.

    The algorithm generates new samples by a smoothed bootstrap approach.
    The generation of new examples corresponds to the generation of data from
    the kernel density estimate of f(x|Y_i), with a smoothing matrix H_j.
    A shrinking matrix can be provided, to set the bandwidth of the gaussian
    kernel.

    Read more in the :ref:`User Guide <rose>`.

    Parameters
    ----------
    {sampling_strategy}
    {random_state}
    shrink_factors : dict of {classes: shrinkfactors} couples, applied to
        the gaussian kernels
    {n_jobs}

    Notes
    -----
    TODO: Support for multi-class resampling. A one-vs.one scheme is used.
    References
    ----------
    .. [1] N. Lunardon, G. Menardi, N.Torelli, "ROSE: A Package for Binary
       Imbalanced Learning," R Journal, 6(1), 2014.

    .. [2] G Menardi, N. Torelli, "Training and assessing classification
       rules with imbalanced data," Data Mining and Knowledge
       Discovery, 28(1), pp.92-122, 2014.
    """

    @_deprecate_positional_args
    def __init__(self, *, sampling_strategy="auto", shrink_factors=None,
                 random_state=None, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.shrink_factors = shrink_factors
        self.n_jobs = n_jobs
        # print("init done: \n {}".format(self.shrink_factors))
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
        # get number of features
        number_of_features = X.shape[1]
        # import random state from API
        random_state = check_random_state(self.random_state)
        # get random subsample of data with replacement
        samples_indices = random_state.choice(
            class_indices, size=n_class_samples, replace=True)
        # compute optimal min(AMISE)
        minimize_amise = (4 / ((number_of_features + 2) * len(
            class_indices))) ** (1 / (number_of_features + 4))
        # create a diagonal matrix with the st.dev. of all classes
        variances = np.std(np.diagflat(X[class_indices, :]),
                           axis=0,
                           ddof=1)
        # compute H_optimal
        h_opt = h_shrink * minimize_amise * variances
        # (sample from multivariate normal)* h_opt + original values
        
        print("""
        inside Rose:
        n_class_sample = {}
        number_of_features = {}
        h_opt = {} , {}
        sample_indices = {}
        """.format(n_class_samples,number_of_features,h_opt, h_opt.shape,samples_indices))
        randoms = np.random.standard_normal(
                        size=(n_class_samples,
                        number_of_features))
        print("randoms = {} , {}".format(randoms, randoms.shape))
        Xrose = randoms @ h_opt + X[samples_indices, :]

        return Xrose

    def _fit_resample(self, X, y):

        # X, y = check_X_y(X, y)
        # X_resampled = np.empty((0, X.shape[1]), dtype=X.dtype)
        # y_resampled = np.empty((0), dtype=X.dtype)

        X_resampled = X.copy()
        y_resampled = y.copy()

        if self.shrink_factors is None:
            self.shrink_factors = {
                key: 0.5 for key in self.sampling_strategy_.keys()
                }
        print(self.shrink_factors)
        for class_sample, n_samples in self.sampling_strategy_.items():
            # get indices of all y's with a given class n
            class_indices = np.flatnonzero(y == class_sample)
            # compute final n. of samples, by n. of elements + n_samples
            n_class_samples = len(class_indices) + n_samples

            print("""
            class_indices = {} \n
            n_class_samples = {} \n
            self.shrink_factors = {}\n
            """.format(class_indices,n_class_samples,self.shrink_factors))
            # resample
            X_new = self._make_samples(X,
                                       class_indices,
                                       n_class_samples,
                                       self.shrink_factors[class_sample])
            y_new = np.array([class_sample] * n_class_samples)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.vstack((X_resampled, X_new))

            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled.astype(X.dtype), y_resampled.astype(y.dtype)
