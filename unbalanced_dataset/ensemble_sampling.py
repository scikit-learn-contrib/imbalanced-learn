from __future__ import print_function
from __future__ import division
from random import sample
import numpy as np
from .unbalanced_dataset import UnbalancedDataset
from .under_sampling import UnderSampler


class EasyEnsemble(UnderSampler):
    """
    Object to perform classification on balanced ensembled selected from
    random sampling.

    It is based on the idea presented in the paper "Exploratory Undersampling
    Class-Imbalance Learning" by Liu et al.
    """

    def __init__(self, ratio=1., random_state=None, replacement=False,
                 n_subsets=10, verbose=True):
        """
        :param ratio:
            The ratio of majority elements to sample with respect to the number
            of minority cases.

        :param random_state:
            Seed.

        :param replacement:
            Either or not to sample randomly with replacement or not.

        :param n_subsets:
            Number of subsets to generate.
        """

        # Passes the relevant parameters back to the parent class.
        UnderSampler.__init__(self, ratio=ratio,
                              random_state=random_state,
                              replacement=replacement,
                              verbose=verbose)

        self.n_subsets = n_subsets

    def resample(self):
        """
        :return subsets_x:
            Python list containing the different data arrays generated and
            balanced.

        :return subsets_y:
            Python list containing the different label arrays generated and
            balanced.
        """

        subsets_x = []
        subsets_y = []

        for s in range(self.n_subsets):
            if self.verbose:
                print("Creation of the set #%i" % s)

            tmp_subset_x, tmp_subset_y = UnderSampler.resample(self)
            subsets_x.append(tmp_subset_x)
            subsets_y.append(tmp_subset_y)

        return subsets_x, subsets_y


class BalanceCascade(UnbalancedDataset):
    """
    Object to perform classification on balanced ensembled selected from
    random sampling and selected using classifier.

    It is based on the idea presented in the paper "Exploratory Undersampling
    Class-Imbalance Learning" by Liu et al.
    """

    def __init__(self, ratio=1., random_state=None, n_max_subset=None,
                 classifier='knn', bootstrap=True, verbose=True, **kwargs):
        """
        :param ratio:
            The ratio of majority elements to sample with respect to the number
            of minority cases.

        :param random_state:
            Seed.

        :param n_max_subset:
            Maximum number of subsets to generate. By default, all data from
            the training will be selected that could lead to a large number of
            subsets. We can probably reduced this number empirically.

        :param classifier:
            The classifier that will be selected to confront the prediction
            with the real labels.

        :param **kwargs:
            The parameters associated with the classifier provided.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state,
                                   verbose=verbose)

        # Define the classifier to use
        if classifier == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            self.classifier = KNeighborsClassifier(**kwargs)
        elif classifier == 'decision-tree':
            from sklearn.tree import DecisionTreeClassifier
            self.classifier = DecisionTreeClassifier(**kwargs)
        elif classifier == 'random-forest':
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(**kwargs)
        elif classifier == 'adaboost':
            from sklearn.ensemble import AdaBoostClassifier
            self.classifier = AdaBoostClassifier(**kwargs)
        elif classifier == 'gradient-boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            self.classifier = GradientBoostingClassifier(**kwargs)
        elif classifier == 'linear-svm':
            from sklearn.svm import LinearSVC
            self.classifier = LinearSVC(**kwargs)
        else:
            raise ValueError('UnbalancedData.BalanceCascade: classifier '
                             'not yet supported.')

        self.n_max_subset = n_max_subset
        self.classifier_name = classifier
        self.bootstrap = bootstrap

    def resample(self):
        """
        :return subsets_x:
            Python list containing the different data arrays generated and
            balanced.

        :return subsets_y:
            Python list containing the different label arrays generated and
            balanced.
        """

        subsets_x = []
        subsets_y = []

        # Start with the minority class
        min_x = self.x[self.y == self.minc]
        min_y = self.y[self.y == self.minc]

        # Condition to initiliase before the search
        b_subset_search = True
        n_subsets = 0
        # Get the initial number of samples to select in the majority class
        n_elt_maj = self.ucd[self.minc]
        # Create the array characterising the array containing the majority
        # class
        N_x = self.x[self.y != self.minc]
        N_y = self.y[self.y != self.minc]
        b_sel_N = np.array([True] * N_y.size)
        idx_mis_class = np.array([])

        # Loop to create the different subsets
        while b_subset_search:
            # Generate an appropriate number of index to extract
            # from the majority class depending of the false classification
            # rate of the previous iteration
            idx_sel_from_maj = np.array(sample(np.nonzero(b_sel_N)[0],
                                               n_elt_maj))
            idx_sel_from_maj = np.concatenate((idx_mis_class,
                                               idx_sel_from_maj),
                                              axis=0).astype(int)

            # Mark these indexes as not being considered for next sampling
            b_sel_N[idx_sel_from_maj] = False

            # For now, we will train and classify on the same data
            # Let see if we should find another solution. Anyway,
            # random stuff are still random stuff
            x_data = np.concatenate((min_x, N_x[idx_sel_from_maj, :]), axis=0)
            y_data = np.concatenate((min_y, N_y[idx_sel_from_maj]), axis=0)

            # Push these data into a new subset
            subsets_x.append(x_data)
            subsets_y.append(y_data)

            if (not ( (self.classifier_name == 'knn'       ) or
                      (self.classifier_name == 'linear-svm')   )
                and self.bootstrap):
                # Apply a bootstrap on x_data
                curr_sample_weight = np.ones((y_data.size,), dtype=np.float64)
                indices = np.random.randint(0, y_data.size, y_data.size)
                sample_counts = np.bincount(indices, minlength=y_data.size)
                curr_sample_weight *= sample_counts

                # Train the classifier using the current data
                self.classifier.fit(x_data, y_data, curr_sample_weight)

            else:
                # Train the classifier using the current data
                self.classifier.fit(x_data, y_data)

            # Predict using only the majority class
            pred_label = self.classifier.predict(N_x[idx_sel_from_maj, :])

            # Basically let's find which sample have to be retained for the
            # next round

            # Find the misclassified index to keep them for the next round
            idx_mis_class = idx_sel_from_maj[np.nonzero(pred_label != N_y[idx_sel_from_maj])]
            if self.verbose:
                print("Elements misclassified: ", idx_mis_class)
            # Count how many random element will be selected
            n_elt_maj = self.ucd[self.minc] - idx_mis_class.size

            if self.verbose:
                print("Creation of the subset #" + str(n_subsets))

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
                    x_data = np.concatenate((min_x, N_x[idx_sel_from_maj, :]), axis=0)
                    y_data = np.concatenate((min_y, N_y[idx_sel_from_maj]), axis=0)
                    # Push these data into a new subset
                    subsets_x.append(x_data)
                    subsets_y.append(y_data)
                    if self.verbose:
                        print("Creation of the subset #" + str(n_subsets))

                        # We found a new subset, increase the counter
                        n_subsets += 1
                    if self.verbose:
                        print('The number of subset achieved their maximum')

            # Also check that we will have enough sample to extract at the
            # next round
            if n_elt_maj > np.count_nonzero(b_sel_N):
                b_subset_search = False
                # Select the remaining data
                idx_sel_from_maj = np.nonzero(b_sel_N)[0]
                idx_sel_from_maj = np.concatenate((idx_mis_class,
                                                   idx_sel_from_maj),
                                                  axis=0).astype(int)
                # Select the final batch
                x_data = np.concatenate((min_x, N_x[idx_sel_from_maj, :]), axis=0)
                y_data = np.concatenate((min_y, N_y[idx_sel_from_maj]), axis=0)
                # Push these data into a new subset
                subsets_x.append(x_data)
                subsets_y.append(y_data)
                if self.verbose:
                    print("Creation of the subset #" + str(n_subsets))

                # We found a new subset, increase the counter
                n_subsets += 1

                if self.verbose:
                    print('Not enough samples to continue creating subsets')

        # Return the different subsets
        return subsets_x, subsets_y
