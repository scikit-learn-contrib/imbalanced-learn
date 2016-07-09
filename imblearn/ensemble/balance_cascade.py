"""Class to perform under-sampling using balace cascade."""
from __future__ import print_function

import numpy as np

from sklearn.utils import check_random_state

from ..base import SamplerMixin


ESTIMATOR_KIND = ('knn', 'decision-tree', 'random-forest', 'adaboost',
                  'gradient-boosting', 'linear-svm')


class BalanceCascade(SamplerMixin):
    """Create an ensemble of balanced sets by iteratively under-sampling the
    imbalanced dataset using an estimator.

    This method iteratively select subset and make an ensemble of the
    different sets. The selection is performed using a specific classifier.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    return_indices : bool, optional (default=True)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    n_max_subset : int or None, optional (default=None)
        Maximum number of subsets to generate. By default, all data from
        the training will be selected that could lead to a large number of
        subsets. We can probably deduce this number empirically.

    classifier : str, optional (default='knn')
        The classifier that will be selected to confront the prediction
        with the real labels. The choices are the following: 'knn',
        'decision-tree', 'random-forest', 'adaboost', 'gradient-boosting'
        and 'linear-svm'.

    bootstrap : bool, optional (default=True)
        Whether to bootstrap the data before each iteration.

    **kwargs : keywords
        The parameters associated with the classifier provided.

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
    The method is described in [1]_.

    This class does not support multi-class.

    References
    ----------
    .. [1] X. Y. Liu, J. Wu and Z. H. Zhou, "Exploratory Undersampling for
       Class-Imbalance Learning," in IEEE Transactions on Systems, Man, and
       Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539-550,
       April 2009.

    """
    def __init__(self, ratio='auto', return_indices=False, random_state=None,
                 n_max_subset=None, classifier='knn', bootstrap=True,
                 **kwargs):
        super(BalanceCascade, self).__init__(ratio=ratio)
        self.return_indices = return_indices
        self.random_state = random_state
        self.classifier = classifier
        self.n_max_subset = n_max_subset
        self.bootstrap = bootstrap
        self.kwargs = kwargs

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
        X_resampled : ndarray, shape (n_subset, n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_subset, n_samples_new)
            The corresponding label of `X_resampled`

        idx_under : ndarray, shape (n_subset, n_samples, )
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """

        if self.classifier not in ESTIMATOR_KIND:
            raise NotImplementedError

        random_state = check_random_state(self.random_state)

        # Define the classifier to use
        if self.classifier == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(
                **self.kwargs)
        elif self.classifier == 'decision-tree':
            from sklearn.tree import DecisionTreeClassifier
            classifier = DecisionTreeClassifier(
                random_state=random_state,
                **self.kwargs)
        elif self.classifier == 'random-forest':
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(
                random_state=random_state,
                **self.kwargs)
        elif self.classifier == 'adaboost':
            from sklearn.ensemble import AdaBoostClassifier
            classifier = AdaBoostClassifier(
                random_state=random_state,
                **self.kwargs)
        elif self.classifier == 'gradient-boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            classifier = GradientBoostingClassifier(
                random_state=random_state,
                **self.kwargs)
        elif self.classifier == 'linear-svm':
            from sklearn.svm import LinearSVC
            classifier = LinearSVC(random_state=random_state,
                                   **self.kwargs)
        else:
            raise NotImplementedError

        X_resampled = []
        y_resampled = []
        if self.return_indices:
            idx_under = []

        # Start with the minority class
        X_min = X[y == self.min_c_]
        y_min = y[y == self.min_c_]

        # Keep the indices of the minority class somewhere if we need to
        # return them later
        if self.return_indices:
            idx_min = np.nonzero(y == self.min_c_)[0]

        # Condition to initiliase before the search
        b_subset_search = True
        n_subsets = 0
        # Get the initial number of samples to select in the majority class
        if self.ratio == 'auto':
            num_samples = self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.stats_c_[self.min_c_] / self.ratio)
        # Create the array characterising the array containing the majority
        # class
        N_x = X[y != self.min_c_]
        N_y = y[y != self.min_c_]
        b_sel_N = np.array([True] * N_y.size)
        idx_mis_class = np.array([])

        # Loop to create the different subsets
        while b_subset_search:
            # Generate an appropriate number of index to extract
            # from the majority class depending of the false classification
            # rate of the previous iteration
            idx_sel_from_maj = random_state.choice(np.nonzero(b_sel_N)[0],
                                                   size=num_samples,
                                                   replace=False)
            idx_sel_from_maj = np.concatenate((idx_mis_class,
                                               idx_sel_from_maj),
                                              axis=0).astype(int)

            # Mark these indexes as not being considered for next sampling
            b_sel_N[idx_sel_from_maj] = False

            # For now, we will train and classify on the same data
            # Let see if we should find another solution. Anyway,
            # random stuff are still random stuff
            x_data = np.concatenate((X_min, N_x[idx_sel_from_maj, :]), axis=0)
            y_data = np.concatenate((y_min, N_y[idx_sel_from_maj]), axis=0)

            # Push these data into a new subset
            X_resampled.append(x_data)
            y_resampled.append(y_data)
            if self.return_indices:
                idx_under.append(np.concatenate((idx_min, idx_sel_from_maj),
                                                axis=0))

            if (not (self.classifier == 'knn' or
                     self.classifier == 'linear-svm') and
                    self.bootstrap):
                # Apply a bootstrap on x_data
                curr_sample_weight = np.ones((y_data.size,), dtype=np.float64)
                indices = random_state.randint(0, y_data.size, y_data.size)
                sample_counts = np.bincount(indices, minlength=y_data.size)
                curr_sample_weight *= sample_counts

                # Train the classifier using the current data
                classifier.fit(x_data, y_data, curr_sample_weight)

            else:
                # Train the classifier using the current data
                classifier.fit(x_data, y_data)

            # Predict using only the majority class
            pred_label = classifier.predict(N_x[idx_sel_from_maj, :])

            # Basically let's find which sample have to be retained for the
            # next round

            # Find the misclassified index to keep them for the next round
            idx_mis_class = idx_sel_from_maj[np.nonzero(pred_label !=
                                                        N_y[idx_sel_from_maj])]
            self.logger.debug('Elements misclassified: %s', idx_mis_class)

            # Count how many random element will be selected
            if self.ratio == 'auto':
                num_samples = self.stats_c_[self.min_c_]
            else:
                num_samples = int(self.stats_c_[self.min_c_] / self.ratio)
            num_samples -= idx_mis_class.size

            self.logger.debug('Creation of the subset #%s', n_subsets)

            # We found a new subset, increase the counter
            n_subsets += 1

            # Check if we have to make an early stopping
            if self.n_max_subset is not None:
                if n_subsets == (self.n_max_subset - 1):
                    b_subset_search = False
                    # Select the remaining data
                    idx_sel_from_maj = np.nonzero(b_sel_N)[0]
                    idx_sel_from_maj = np.concatenate((idx_mis_class,
                                                       idx_sel_from_maj),
                                                      axis=0).astype(int)
                    # Select the final batch
                    x_data = np.concatenate((X_min, N_x[idx_sel_from_maj, :]),
                                            axis=0)
                    y_data = np.concatenate((y_min, N_y[idx_sel_from_maj]),
                                            axis=0)
                    # Push these data into a new subset
                    X_resampled.append(x_data)
                    y_resampled.append(y_data)
                    if self.return_indices:
                        idx_under.append(np.concatenate((idx_min,
                                                         idx_sel_from_maj),
                                                        axis=0))

                    self.logger.debug('Creation of the subset #%s', n_subsets)

                    # We found a new subset, increase the counter
                    n_subsets += 1

                    self.logger.debug('The number of subset reached is'
                                      ' maximum.')

            # Also check that we will have enough sample to extract at the
            # next round
            if num_samples > np.count_nonzero(b_sel_N):
                b_subset_search = False
                # Select the remaining data
                idx_sel_from_maj = np.nonzero(b_sel_N)[0]
                idx_sel_from_maj = np.concatenate((idx_mis_class,
                                                   idx_sel_from_maj),
                                                  axis=0).astype(int)
                # Select the final batch
                x_data = np.concatenate((X_min, N_x[idx_sel_from_maj, :]),
                                        axis=0)
                y_data = np.concatenate((y_min, N_y[idx_sel_from_maj]), axis=0)
                # Push these data into a new subset
                X_resampled.append(x_data)
                y_resampled.append(y_data)
                if self.return_indices:
                    idx_under.append(np.concatenate((idx_min,
                                                     idx_sel_from_maj),
                                                    axis=0))
                self.logger.debug('Creation of the subset #%s', n_subsets)

                # We found a new subset, increase the counter
                n_subsets += 1

                self.logger.debug('Not enough samples to continue creating'
                                  ' subsets.')

        if self.return_indices:
            return (np.array(X_resampled), np.array(y_resampled),
                    np.array(idx_under))
        else:
            return np.array(X_resampled), np.array(y_resampled)
