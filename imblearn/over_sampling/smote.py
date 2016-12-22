"""Class to perform over-sampling using SMOTE."""
from __future__ import division, print_function

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.base import KNeighborsMixin
from sklearn.svm import SVC
from sklearn.utils import check_array, check_random_state

from ..base import BaseBinarySampler

SMOTE_KIND = ('regular', 'borderline1', 'borderline2', 'svm')


class SMOTE(BaseBinarySampler):
    """Class to perform over-sampling using SMOTE.

    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique, and the variants Borderline SMOTE 1, 2 and
    SVM-SMOTE.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    k : int, optional (default=None)
        Number of nearest neighbours to used to construct synthetic samples.

        NOTE: `k` is deprecated from 0.2 and will be replaced in 0.4
        Use ``k_neighbors`` instead.

    k_neighbors : int or object, optional (default=5)
        If int, number of nearest neighbours to used to construct
        synthetic samples.
        If object, an estimator that inherits from
        `sklearn.neighbors.base.KNeighborsMixin` that will be used to find
        the k_neighbors.

    m : int, optional (default=None)
        Number of nearest neighbours to use to determine if a minority sample
        is in danger.

        NOTE: `m` is deprecated from 0.2 and will be replaced in 0.4
        Use ``m_neighbors`` instead.

    m_neighbors : int int or object, optional (default=10)
        If int, number of nearest neighbours to use to determine if a minority
        sample is in danger.
        If object, an estimator that inherits from
        `sklearn.neighbors.base.KNeighborsMixin` that will be used to find
        the k_neighbors.

    out_step : float, optional (default=0.5)
        Step size when extrapolating.

    kind : str, optional (default='regular')
        The type of SMOTE algorithm to use one of the following options:
        'regular', 'borderline1', 'borderline2', 'svm'.

    svm_estimator : object, optional (default=SVC())
        If `kind='svm'`, a parametrized `sklearn.svm.SVC` classifier can
        be passed.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    See the original papers: [1]_, [2]_, [3]_ for more details.

    It does not support multiple classes automatically, but can be called
    multiple times.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
    SMOTE # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> sm = SMOTE(random_state=42)
    >>> X_res, y_res = sm.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 900, 1: 900})

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    .. [2] H. Han, W. Wen-Yuan, M. Bing-Huan, "Borderline-SMOTE: a new
       over-sampling method in imbalanced data sets learning," Advances in
       intelligent computing, 878-887, 2005.

    .. [3] H. M. Nguyen, E. W. Cooper, K. Kamei, "Borderline over-sampling for
       imbalanced data classification," International Journal of Knowledge
       Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2001.

    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 k=None,
                 k_neighbors=5,
                 m=None,
                 m_neighbors=10,
                 out_step=0.5,
                 kind='regular',
                 svm_estimator=None,
                 n_jobs=1):
        super(SMOTE, self).__init__(ratio=ratio, random_state=random_state)
        self.kind = kind
        self.k = k
        self.k_neighbors = k_neighbors
        self.m = m
        self.m_neighbors = m_neighbors
        self.out_step = out_step
        self.svm_estimator = svm_estimator
        self.n_jobs = n_jobs

    def _in_danger_noise(self, samples, y, kind='danger'):
        """Estimate if a set of sample are in danger or noise.

        Parameters
        ----------
        samples : ndarray, shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        y : ndarray, shape (n_samples, )
            The true label in order to check the neighbour labels.

        kind : str, optional (default='danger')
            The type of classification to use. Can be either:

            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        Returns
        -------
        output : ndarray, shape (n_samples, )
            A boolean array where True refer to samples in danger or noise.

        """

        # Find the NN for each samples
        # Exclude the sample itself
        x = self.nn_m_.kneighbors(samples, return_distance=False)[:, 1:]

        # Count how many NN belong to the minority class
        # Find the class corresponding to the label in x
        nn_label = (y[x] != self.min_c_).astype(int)
        # Compute the number of majority samples in the NN
        n_maj = np.sum(nn_label, axis=1)

        if kind == 'danger':
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(
                n_maj >= float(self.nn_m_.n_neighbors - 1) / 2.,
                n_maj < self.nn_m_.n_neighbors - 1)
        elif kind == 'noise':
            # Samples are noise for m = m'
            return n_maj == self.nn_m_.n_neighbors - 1
        else:
            raise NotImplementedError

    def _make_samples(self,
                      X,
                      y_type,
                      nn_data,
                      nn_num,
                      n_samples,
                      step_size=1.):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Points from which the points will be created.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in nn_data.

        n_samples : int
            The number of samples to generate.

        step_size : float, optional (default=1.)
            The step size to create samples.

        Returns
        -------
        X_new : ndarray, shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray, shape (n_samples_new, )
            Target values for synthetic samples.

        """

        # Check the consistency of X
        X = check_array(X)
        # Check the random state
        random_state = check_random_state(self.random_state)

        # A matrix to store the synthetic samples
        X_new = np.zeros((n_samples, X.shape[1]))

        # # Set seeds
        # seeds = random_state.randint(low=0,
        #                              high=100 * len(nn_num.flatten()),
        #                              size=n_samples)

        # Randomly pick samples to construct neighbours from
        samples = random_state.randint(
            low=0, high=len(nn_num.flatten()), size=n_samples)

        # Loop over the NN matrix and create new samples
        for i, n in enumerate(samples):
            # NN lines relate to original sample, columns to its
            # nearest neighbours
            row, col = divmod(n, nn_num.shape[1])

            # Take a step of random size (0,1) in the direction of the
            # n nearest neighbours
            # if self.random_state is None:
            #     np.random.seed(seeds[i])
            # else:
            #     np.random.seed(self.random_state)
            step = step_size * random_state.uniform()

            # Construct synthetic sample
            X_new[i] = X[row] - step * (X[row] - nn_data[nn_num[row, col]])

        # The returned target vector is simply a repetition of the
        # minority label
        y_new = np.array([y_type] * len(X_new))

        self.logger.info('Generated %s new samples ...', len(X_new))

        return X_new, y_new

    def _validate_estimator(self):
        # --- NN object
        # Import the NN object from scikit-learn library. Since in the smote
        # variations we must first find samples that are in danger, we
        # initialize the NN object differently depending on the method chosen
        if self.kind == 'regular':
            # Regular smote does not look for samples in danger, instead it
            # creates synthetic samples directly from the k-th nearest
            # neighbours with not filtering
            if isinstance(self.k_neighbors, int):
                self.nn_k_ = NearestNeighbors(
                    n_neighbors=self.k_neighbors + 1, n_jobs=self.n_jobs)
            elif isinstance(self.k_neighbors, KNeighborsMixin):
                self.nn_k_ = self.k_neighbors
            else:
                raise ValueError('`n_neighbors` has to be be either int or a'
                                 ' subclass of KNeighborsMixin.')
        else:
            # Borderline1, 2 and SVM variations of smote must first look for
            # samples that could be considered noise and samples that live
            # near the boundary between the classes. Therefore, before
            # creating synthetic samples from the k-th nns, it first look
            # for m nearest neighbors to decide whether or not a sample is
            # noise or near the boundary.
            if isinstance(self.k_neighbors, int):
                self.nn_k_ = NearestNeighbors(
                    n_neighbors=self.k_neighbors + 1, n_jobs=self.n_jobs)
            elif isinstance(self.k_neighbors, KNeighborsMixin):
                self.nn_k_ = self.k_neighbors
            else:
                raise ValueError('`n_neighbors` has to be be either int or a'
                                 ' subclass of KNeighborsMixin.')

            if isinstance(self.m_neighbors, int):
                self.nn_m_ = NearestNeighbors(
                    n_neighbors=self.m_neighbors + 1, n_jobs=self.n_jobs)
            elif isinstance(self.m_neighbors, KNeighborsMixin):
                self.nn_m_ = self.m_neighbors
            else:
                raise ValueError('`n_neighbors` has to be be either int or a'
                                 ' subclass of KNeighborsMixin.')

        # --- SVM smote
        # Unlike the borderline variations, the SVM variation uses the support
        # vectors to decide which samples are in danger (near the boundary).
        # Additionally it also introduces extrapolation for samples that are
        # considered safe (far from boundary) and interpolation for samples
        # in danger (near the boundary). The level of extrapolation is
        # controled by the out_step.
        if self.kind == 'svm':
            if self.svm_estimator is None:
                # Store SVM object with any parameters
                self.svm_estimator_ = SVC(random_state=self.random_state)
            elif isinstance(self.svm_estimator, SVC):
                self.svm_estimator_ = self.svm_estimator
            else:
                raise ValueError('`svm_estimator` has to be an SVC object')

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """

        super(SMOTE, self).fit(X, y)

        self._validate_estimator()

        return self

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """

        if self.kind not in SMOTE_KIND:
            raise ValueError('Unknown kind for SMOTE algorithm.')

        random_state = check_random_state(self.random_state)

        # Define the number of sample to create
        # We handle only two classes problem for the moment.
        if self.ratio == 'auto':
            num_samples = (
                self.stats_c_[self.maj_c_] - self.stats_c_[self.min_c_])
        else:
            num_samples = int((self.ratio * self.stats_c_[self.maj_c_]) -
                              self.stats_c_[self.min_c_])

        # Start by separating minority class features and target values.
        X_min = X[y == self.min_c_]

        # If regular SMOTE is to be performed
        if self.kind == 'regular':

            self.logger.debug('Finding the %s nearest neighbours ...',
                              self.nn_k_.n_neighbors - 1)

            # Look for k-th nearest neighbours, excluding, of course, the
            # point itself.
            self.nn_k_.fit(X_min)

            # Matrix with k-th nearest neighbours indexes for each minority
            # element.
            nns = self.nn_k_.kneighbors(X_min, return_distance=False)[:, 1:]

            self.logger.debug('Create synthetic samples ...')

            # --- Generating synthetic samples
            # Use static method make_samples to generate minority samples
            X_new, y_new = self._make_samples(X_min, self.min_c_, X_min, nns,
                                              num_samples, 1.0)

            # Concatenate the newly generated samples to the original data set
            X_resampled = np.concatenate((X, X_new), axis=0)
            y_resampled = np.concatenate((y, y_new), axis=0)

            return X_resampled, y_resampled

        if self.kind == 'borderline1' or self.kind == 'borderline2':

            self.logger.debug('Finding the %s nearest neighbours ...',
                              self.nn_m_.n_neighbors - 1)

            # Find the NNs for all samples in the data set.
            self.nn_m_.fit(X)

            # Boolean array with True for minority samples in danger
            danger_index = self._in_danger_noise(X_min, y, kind='danger')

            # If all minority samples are safe, return the original data set.
            if not any(danger_index):
                self.logger.debug('There are no samples in danger. No'
                                  ' borderline synthetic samples created.')

                # All are safe, nothing to be done here.
                return X, y

            # If we got here is because some samples are in danger, we need to
            # find the NNs among the minority class to create the new synthetic
            # samples.
            #
            # We start by changing the number of NNs to consider from m + 1
            # to k + 1
            self.nn_k_.fit(X_min)

            # nns...#
            nns = self.nn_k_.kneighbors(
                X_min[danger_index], return_distance=False)[:, 1:]

            # B1 and B2 types diverge here!!!
            if self.kind == 'borderline1':
                # Create synthetic samples for borderline points.
                X_new, y_new = self._make_samples(
                    X_min[danger_index], self.min_c_, X_min, nns, num_samples)

                # Concatenate the newly generated samples to the original
                # dataset
                X_resampled = np.concatenate((X, X_new), axis=0)
                y_resampled = np.concatenate((y, y_new), axis=0)

                return X_resampled, y_resampled

            else:
                # Split the number of synthetic samples between only minority
                # (type 1), or minority and majority (with reduced step size)
                # (type 2).
                # The fraction is sampled from a beta distribution centered
                # around 0.5 with variance ~0.01
                fractions = random_state.beta(10, 10)

                # Only minority
                X_new_1, y_new_1 = self._make_samples(
                    X_min[danger_index],
                    self.min_c_,
                    X_min,
                    nns,
                    int(fractions * (num_samples + 1)),
                    step_size=1.)

                # Only majority with smaller step size
                X_new_2, y_new_2 = self._make_samples(
                    X_min[danger_index],
                    self.min_c_,
                    X[y != self.min_c_],
                    nns,
                    int((1 - fractions) * num_samples),
                    step_size=0.5)

                # Concatenate the newly generated samples to the original
                # data set
                X_resampled = np.concatenate((X, X_new_1, X_new_2), axis=0)
                y_resampled = np.concatenate((y, y_new_1, y_new_2), axis=0)

                return X_resampled, y_resampled

        if self.kind == 'svm':
            # The SVM smote model fits a support vector machine
            # classifier to the data and uses the support vector to
            # provide a notion of boundary. Unlike regular smote, where
            # such notion relies on proportion of nearest neighbours
            # belonging to each class.

            # Fit SVM to the full data#
            self.svm_estimator_.fit(X, y)

            # Find the support vectors and their corresponding indexes
            support_index = self.svm_estimator_.support_[y[
                self.svm_estimator_.support_] == self.min_c_]
            support_vector = X[support_index]

            # First, find the nn of all the samples to identify samples
            # in danger and noisy ones
            self.logger.debug('Finding the %s nearest neighbours ...',
                              self.nn_m_.n_neighbors - 1)

            # As usual, fit a nearest neighbour model to the data
            self.nn_m_.fit(X)

            # Now, get rid of noisy support vectors
            noise_bool = self._in_danger_noise(support_vector, y, kind='noise')

            # Remove noisy support vectors
            support_vector = support_vector[np.logical_not(noise_bool)]
            danger_bool = self._in_danger_noise(
                support_vector, y, kind='danger')
            safety_bool = np.logical_not(danger_bool)

            self.logger.debug('Out of %s support vectors, %s are noisy, '
                              '%s are in danger '
                              'and %s are safe.', support_vector.shape[0],
                              noise_bool.sum().astype(int),
                              danger_bool.sum().astype(int),
                              safety_bool.sum().astype(int))

            # Proceed to find support vectors NNs among the minority class
            self.logger.debug('Finding the %s nearest neighbours ...',
                              self.nn_k_.n_neighbors - 1)

            self.nn_k_.fit(X_min)

            self.logger.debug('Create synthetic samples ...')

            # Split the number of synthetic samples between interpolation and
            # extrapolation

            # The fraction are sampled from a beta distribution with mean
            # 0.5 and variance 0.01#
            fractions = random_state.beta(10, 10)

            # Interpolate samples in danger
            if np.count_nonzero(danger_bool) > 0:
                nns = self.nn_k_.kneighbors(
                    support_vector[danger_bool], return_distance=False)[:, 1:]

                X_new_1, y_new_1 = self._make_samples(
                    support_vector[danger_bool],
                    self.min_c_,
                    X_min,
                    nns,
                    int(fractions * (num_samples + 1)),
                    step_size=1.)

            # Extrapolate safe samples
            if np.count_nonzero(safety_bool) > 0:
                nns = self.nn_k_.kneighbors(
                    support_vector[safety_bool], return_distance=False)[:, 1:]

                X_new_2, y_new_2 = self._make_samples(
                    support_vector[safety_bool],
                    self.min_c_,
                    X_min,
                    nns,
                    int((1 - fractions) * num_samples),
                    step_size=-self.out_step)

            # Concatenate the newly generated samples to the original data set
            if (np.count_nonzero(danger_bool) > 0 and
                    np.count_nonzero(safety_bool) > 0):
                X_resampled = np.concatenate((X, X_new_1, X_new_2), axis=0)
                y_resampled = np.concatenate((y, y_new_1, y_new_2), axis=0)
            # not any support vectors in danger
            elif np.count_nonzero(danger_bool) == 0:
                X_resampled = np.concatenate((X, X_new_2), axis=0)
                y_resampled = np.concatenate((y, y_new_2), axis=0)
            # All the support vector in danger
            elif np.count_nonzero(safety_bool) == 0:
                X_resampled = np.concatenate((X, X_new_1), axis=0)
                y_resampled = np.concatenate((y, y_new_1), axis=0)

            return X_resampled, y_resampled
