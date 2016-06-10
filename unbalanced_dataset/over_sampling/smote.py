"""Class to perform over-sampling using SMOTE."""
from __future__ import print_function
from __future__ import division

import numpy as np

from numpy.random import beta

from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from .over_sampler import OverSampler


class SMOTE(OverSampler):
    """Class to perform over-sampling using SMOTE.

    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique, and the variations Borderline SMOTE 1, 2 and
    SVM-SMOTE.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balanced
        the dataset. Otherwise, the ratio will corresponds to the number
        of samples in the minority class over the the number of samples
        in the majority class.

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Boolean to either or not print information about the processing.

    k : int, optional (default=5)
        Number of nearest neighbours to used to construct synthetic samples.

    m : int, optional (default=10)
        Number of nearest neighbours to use to determine if a minority sample
        is in danger.

    out_step : float, optional (default=0.5)
        Step size when extrapolating.

    kind : str, optional (default='regular')
        The type of SMOTE algorithm to use one of the following options:
        'regular', 'borderline1', 'borderline2', 'svm'

    Attributes
    ----------
    ratio_ : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balanced
        the dataset. Otherwise, the ratio will corresponds to the number
        of samples in the minority class over the the number of samples
        in the majority class.

    rs_ : int or None, optional (default=None)
        Seed for random number generation.

    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    Notes
    -----
    See the original papers: [1]_, [2]_, [3]_ for more details.

    It does not support multiple classes automatically, but can be called
    multiple times.

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

    def __init__(self, ratio='auto', random_state=None, verbose=True,
                 k=5, m=10, out_step=0.5, kind='regular', n_jobs=-1, **kwargs):
        """Initialisation of SMOTE object.

        Parameters
        ----------
        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. Otherwise, the ratio will corresponds to the
            number of samples in the minority class over the the number of
            samples in the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Boolean to either or not print information about the
            processing.

        k : int, optional (default=5)
            Number of nearest neighbours to used to construct synthetic
            samples.

        m : int, optional (default=10)
            Number of nearest neighbours to use to determine if a minority
            sample is in danger.

        out_step : float, optional (default=0.5)
            Step size when extrapolating.

        kind : str, optional (default='regular')
            The type of SMOTE algorithm to use one of the following
            options: 'regular', 'borderline1', 'borderline2', 'svm'

        n_jobs : int, optional (default=-1)
            Number of threads to run the algorithm when it is possible.

        """
        super(SMOTE, self).__init__(ratio=ratio,
                                    random_state=random_state,
                                    verbose=verbose)

        # Check the number of thread to use
        self.n_jobs = n_jobs

        # --- The type of smote
        # This object can perform regular smote over-sampling, borderline 1,
        # borderline 2 and svm smote. Since the algorithms are fairly simple
        # they share most methods.
        possible_kind = ('regular', 'borderline1', 'borderline2', 'svm')
        if kind in possible_kind:
            self.kind = kind
        else:
            raise ValueError('Unknown kind for SMOTE algorithm.')

        # --- Verbose
        # Control whether or not status and progress information should be
        self.verbose = verbose

        # --- Nearest Neighbours for synthetic samples
        # The smote algorithm uses the k-th nearest neighbours of a minority
        # sample to generate new synthetic samples.
        self.k = k

        # --- NN object
        # Import the NN object from scikit-learn library. Since in the smote
        # variations we must first find samples that are in danger, we
        # initialize the NN object differently depending on the method chosen
        if kind == 'regular':
            # Regular smote does not look for samples in danger, instead it
            # creates synthetic samples directly from the k-th nearest
            # neighbours with not filtering
            self.nearest_neighbour_ = NearestNeighbors(n_neighbors=k + 1,
                                                           n_jobs=self.n_jobs)
        else:
            # Borderline1, 2 and SVM variations of smote must first look for
            # samples that could be considered noise and samples that live
            # near the boundary between the classes. Therefore, before
            # creating synthetic samples from the k-th nns, it first look
            # for m nearest neighbors to decide whether or not a sample is
            # noise or near the boundary.
            self.nearest_neighbour_ = NearestNeighbors(n_neighbors=m + 1,
                                                           n_jobs=self.n_jobs)

            # --- Nearest Neighbours for noise and boundary (in danger)
            # Before creating synthetic samples we must first decide if
            # a given entry is noise or in danger. We use m nns in this step
            self.m = m

        # --- SVM smote
        # Unlike the borderline variations, the SVM variation uses the support
        # vectors to decide which samples are in danger (near the boundary).
        # Additionally it also introduces extrapolation for samples that are
        # considered safe (far from boundary) and interpolation for samples
        # in danger (near the boundary). The level of extrapolation is
        # controled by the out_step.
        if kind == 'svm':
            # Store extrapolation size
            self.out_step = out_step

            # Store SVM object with any parameters
            self.svm_ = SVC(random_state=self.rs_, **kwargs)

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
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        # Call the parent function
        super(SMOTE, self).fit(X, y)

        return self

    def _in_danger_noise(self, samples, y, kind='danger'):
        """Estimate if a set of sample are in danger or not.

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
        x = self.nearest_neighbour_.kneighbors(samples,
                                               return_distance=False)[:, 1:]

        # Count how many NN belong to the minority class
        # Find the class corresponding to the label in x
        nn_label = (y[x] != self.min_c_).astype(int)
        # Compute the number of majority samples in the NN
        n_maj = np.sum(nn_label, axis=1)

        if kind == 'danger':
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(n_maj >= float(self.m) / 2.,
                                  n_maj < self.m)
        elif kind == 'noise':
            # Samples are noise for m = m'
            return n_maj == self.m
        else:
            raise NotImplementedError

    def _make_samples(self, X, y_type, nn_data, nn_num, n_samples,
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

        # A matrix to store the synthetic samples
        X_new = np.zeros((n_samples, X.shape[1]))

        # Set seeds
        np.random.seed(self.rs_)
        seeds = np.random.randint(low=0,
                                  high=100*len(nn_num.flatten()),
                                  size=n_samples)

        # Randomly pick samples to construct neighbours from
        np.random.seed(self.rs_)
        samples = np.random.randint(low=0,
                                    high=len(nn_num.flatten()),
                                    size=n_samples)

        # Loop over the NN matrix and create new samples
        for i, n in enumerate(samples):
            # NN lines relate to original sample, columns to its
            # nearest neighbours
            row, col = divmod(n, nn_num.shape[1])

            # Take a step of random size (0,1) in the direction of the
            # n nearest neighbours
            if self.rs_ is None:
                np.random.seed(seeds[i])
            else:
                np.random.seed(self.rs_)
            step = step_size * np.random.uniform()

            # Construct synthetic sample
            X_new[i] = X[row] - step * (X[row] -
                                        nn_data[nn_num[row, col]])

        # The returned target vector is simply a repetition of the
        # minority label
        y_new = np.array([y_type] * len(X_new))

        if self.verbose:
            print("Generated {} new samples ...".format(len(X_new)))

        return X_new, y_new

    def transform(self, X, y):
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
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        # Call the parent function
        super(SMOTE, self).transform(X, y)

        # Define the number of sample to create
        # We handle only two classes problem for the moment.
        if self.ratio_ == 'auto':
            num_samples = (self.stats_c_[self.maj_c_] -
                           self.stats_c_[self.min_c_])
        else:
            num_samples = ((self.ratio_ * self.stats_c_[self.maj_c_]) -
                           self.stats_c_[self.min_c_])

        # Start by separating minority class features and target values.
        X_min = X[y == self.min_c_]

        # If regular SMOTE is to be performed
        if self.kind == 'regular':

            # Print if verbose is true#
            if self.verbose:
                print('Finding the {} nearest neighbours...'.format(self.k))

            # Look for k-th nearest neighbours, excluding, of course, the
            # point itself.
            self.nearest_neighbour_.fit(X_min)

            # Matrix with k-th nearest neighbours indexes for each minority
            # element.
            nns = self.nearest_neighbour_.kneighbors(
                X_min,
                return_distance=False)[:, 1:]

            # Print status if verbose is true
            if self.verbose:
                print("done!")
                print("Creating synthetic samples...", end="")

            # --- Generating synthetic samples
            # Use static method make_samples to generate minority samples
            X_new, y_new = self._make_samples(X_min,
                                             self.min_c_,
                                             X_min,
                                             nns,
                                             num_samples,
                                             1.0)

            if self.verbose:
                print("done!")

            # Concatenate the newly generated samples to the original data set
            X_resampled = np.concatenate((X, X_new), axis=0)
            y_resampled = np.concatenate((y, y_new), axis=0)

            return X_resampled, y_resampled

        if self.kind == 'borderline1' or self.kind == 'borderline2':

            if self.verbose:
                print("Finding the {} nearest neighbours...".format(self.m))

            # Find the NNs for all samples in the data set.
            self.nearest_neighbour_.fit(X)

            if self.verbose:
                print("done!")

            # Boolean array with True for minority samples in danger
            danger_index = self._in_danger_noise(X_min, y, kind='danger')

            # If all minority samples are safe, return the original data set.
            if not any(danger_index):
                if self.verbose:
                    print('There are no samples in danger. No borderline '
                          'synthetic samples created.')

                # All are safe, nothing to be done here.
                return X, y

            # If we got here is because some samples are in danger, we need to
            # find the NNs among the minority class to create the new synthetic
            # samples.
            #
            # We start by changing the number of NNs to consider from m + 1
            # to k + 1
            self.nearest_neighbour_.set_params(**{'n_neighbors': self.k + 1})
            self.nearest_neighbour_.fit(X_min)

            # nns...#
            nns = self.nearest_neighbour_.kneighbors(
                X_min[danger_index],
                return_distance=False)[:, 1:]

            # B1 and B2 types diverge here!!!
            if self.kind == 'borderline1':
                # Create synthetic samples for borderline points.
                X_new, y_new = self._make_samples(X_min[danger_index],
                                                 self.min_c_,
                                                 X_min,
                                                 nns,
                                                 num_samples)

                # Concatenate the newly generated samples to the original
                # dataset
                X_resampled = np.concatenate((X, X_new), axis=0)
                y_resampled = np.concatenate((y, y_new), axis=0)

                # Reset the k-neighbours to m+1 neighbours
                self.nearest_neighbour_.set_params(**{'n_neighbors': self.m+1})

                return X_resampled, y_resampled

            else:
                # Split the number of synthetic samples between only minority
                # (type 1), or minority and majority (with reduced step size)
                # (type 2).
                np.random.seed(self.rs_)
                # The fraction is sampled from a beta distribution centered
                # around 0.5 with variance ~0.01
                fractions = beta(10, 10)

                # Only minority
                X_new_1, y_new_1 = self._make_samples(X_min[danger_index],
                                                     self.min_c_,
                                                     X_min,
                                                     nns,
                                                     int(fractions *
                                                         (num_samples + 1)),
                                                     step_size=1.)

                # Only majority with smaller step size
                X_new_2, y_new_2 = self._make_samples(X_min[danger_index],
                                                     self.min_c_,
                                                     X[y != self.min_c_],
                                                     nns,
                                                     int((1 - fractions) *
                                                         num_samples),
                                                     step_size=0.5)

                # Concatenate the newly generated samples to the original
                # data set
                X_resampled = np.concatenate((X, X_new_1, X_new_2), axis=0)
                y_resampled = np.concatenate((y, y_new_1, y_new_2), axis=0)

                # Reset the k-neighbours to m+1 neighbours
                self.nearest_neighbour_.set_params(**{'n_neighbors': self.m+1})

                return X_resampled, y_resampled

        if self.kind == 'svm':
            # The SVM smote model fits a support vector machine
            # classifier to the data and uses the support vector to
            # provide a notion of boundary. Unlike regular smote, where
            # such notion relies on proportion of nearest neighbours
            # belonging to each class.

            # Fit SVM to the full data#
            self.svm_.fit(X, y)

            # Find the support vectors and their corresponding indexes
            support_index = self.svm_.support_[y[self.svm_.support_] ==
                                               self.min_c_]
            support_vector = X[support_index]

            # First, find the nn of all the samples to identify samples
            # in danger and noisy ones
            if self.verbose:
                print("Finding the {} nearest neighbours...".format(self.m))

            # As usual, fit a nearest neighbour model to the data
            self.nearest_neighbour_.fit(X)

            if self.verbose:
                print("done!")

            # Now, get rid of noisy support vectors

            noise_bool = self._in_danger_noise(support_vector, y, kind='noise')

            # Remove noisy support vectors
            support_vector = support_vector[np.logical_not(noise_bool)]
            danger_bool = self._in_danger_noise(support_vector, y,
                                               kind='danger')
            safety_bool = np.logical_not(danger_bool)

            if self.verbose:
                print("Out of {0} support vectors, {1} are noisy, "
                      "{2} are in danger "
                      "and {3} are safe.".format(support_vector.shape[0],
                                                 noise_bool.sum().astype(int),
                                                 danger_bool.sum().astype(int),
                                                 safety_bool.sum().astype(int)
                                                 ))

                # Proceed to find support vectors NNs among the minority class
                print("Finding the {} nearest neighbours...".format(self.k))

            self.nearest_neighbour_.set_params(**{'n_neighbors': self.k + 1})
            self.nearest_neighbour_.fit(X_min)

            if self.verbose:
                print("done!")
                print("Creating synthetic samples...", end="")

            # Split the number of synthetic samples between interpolation and
            # extrapolation

            # The fraction are sampled from a beta distribution with mean
            # 0.5 and variance 0.01#
            np.random.seed(self.rs_)
            fractions = beta(10, 10)

            # Interpolate samples in danger
            if np.count_nonzero(danger_bool) > 0:
                nns = self.nearest_neighbour_.kneighbors(
                    support_vector[danger_bool],
                    return_distance=False)[:, 1:]

                X_new_1, y_new_1 = self._make_samples(
                    support_vector[danger_bool],
                    self.min_c_,
                    X_min,
                    nns,
                    int(fractions * (num_samples + 1)),
                    step_size=1.)

            # Extrapolate safe samples
            if np.count_nonzero(safety_bool) > 0:
                nns = self.nearest_neighbour_.kneighbors(
                    support_vector[safety_bool],
                    return_distance=False)[:, 1:]

                X_new_2, y_new_2 = self._make_samples(
                    support_vector[safety_bool],
                    self.min_c_,
                    X_min,
                    nns,
                    int((1 - fractions) * num_samples),
                    step_size=-self.out_step)

            if self.verbose:
                print("done!")

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

            # Reset the k-neighbours to m+1 neighbours
            self.nearest_neighbour_.set_params(**{'n_neighbors': self.m+1})

            return X_resampled, y_resampled
