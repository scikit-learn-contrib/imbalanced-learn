"""Class to perfrom over-sampling using MLSMOTE."""

import itertools
import numpy as np
from scipy import sparse

from sklearn.utils import check_random_state


class MLSMOTE:
    """Over-sampling using MLSMOTE.

    Parameters
    ----------
    sampling_strategy: 'ranking', 'union' or 'intersection' default: 'ranking'
        Strategy to generate labelsets


    k_neighbors : int or object, default=5
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.

    categorical_features : ndarray of shape (n_cat_features,) or (n_features,)
        Specifies which features are categorical. Can either be:

        - array of indices specifying the categorical features;
        - mask array of shape (n_features, ) and ``bool`` dtype for which
          ``True`` indicates the categorical features.

    Notes
    -----
    The implementation is based on [1]_.

    References
    ----------
    .. [1] Charte, F. & Rivera Rivas, Antonio & Del Jesus, María José & Herrera,
           Francisco. (2015). "MLSMOTE: Approaching imbalanced multilabel learning
           through synthetic instance generation."
           Knowledge-Based Systems. -. 10.1016/j.knosys.2015.07.019.

    Examples
    --------
    >>> from sklearn.datasets import make_multilabel_classification
    """

    def __init__(
        self,
        *,
        sampling_strategy="ranking",
        categorical_features,
        random_state=None,
        k_neighbors=5,
    ):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.sampling_strategy_ = sampling_strategy
        self.categorical_features = categorical_features
        self.continuous_features_ = None
        self.features = []

    def fit_resample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : {array-like, sparse matrix of shape \
                (n_samples, n_labels)
            or a list of lists of labels.
            See "sklearn.datasets.make_multilabel_classification" and \
                the "return_indicate" input parameter for more \
                information on possible label sets formats.

            Corresponding label sets for each sample in X. Sparse matrix \
                should be of CSR format.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new, n_labels)
            The corresponding label sets of `X_resampled`.
        """
        self.n_features_ = X.shape[1]

        self._validate_estimator()
        random_state = check_random_state(self.random_state)

        X_resampled = X.copy()
        y_resampled = y.copy()

        if type(y) == np.ndarray or type(y) == sparse._csr.csr_matrix:
            labels = y
            unique_labels = range(0, y.shape[1])
        elif type(y) == list:
            labels = np.array([np.array(xi) for xi in y], dtype=object)
            unique_labels = self._collect_unique_labels(y)
        else:
            raise TypeError(
                "'y' can only be of type 'numpy.ndarray', 'scipy.sparse._csr.csr_matrix'"
                " or 'list'"
            )
        self.features = X

        X_synth = []
        y_synth = []

        append_X_synth = X_synth.append
        append_y_synth = y_synth.append
        mean_ir = self._get_mean_imbalance_ratio(unique_labels, labels)

        if sparse.issparse(y):
            y_synth = None

            for label in unique_labels:
                irlbl_num = self._get_imbalance_ratio_numerator(
                    unique_labels, y_resampled
                )
                irlbl = self._get_imbalance_ratio_per_label(
                    label, irlbl_num, y_resampled
                )
                if irlbl > mean_ir:
                    min_bag = self._get_all_instances_of_label(label, labels)
                    for sample in min_bag:
                        distances = self._calc_distances(
                            sample, min_bag, unique_labels, labels
                        )
                        distances = np.sort(distances, order="distance")
                        neighbours = distances[: self.k_neighbors]
                        ref_neigh = random_state.choice(neighbours, 1)[0]
                        X_new, y_new = self._create_new_sample(
                            sample,
                            ref_neigh[1],
                            [x[1] for x in neighbours],
                            unique_labels,
                            labels,
                            random_state,
                        )
                        append_X_synth(X_new)
                        y_resambled = sparse.vstack((y_resampled, y_new))
            return np.concatenate((X_resampled, np.array(X_synth))), y_resampled
        else:
            for index, label in np.ndenumerate(unique_labels):
                irlbl_num = self._get_imbalance_ratio_numerator(
                    unique_labels, y_resampled
                )
                irlbl = self._get_imbalance_ratio_per_label(
                    label, irlbl_num, y_resampled
                )
                if irlbl > mean_ir:
                    min_bag = self._get_all_instances_of_label(label, labels)
                    for sample in min_bag:
                        distances = self._calc_distances(
                            sample, min_bag, unique_labels, labels
                        )
                        distances = np.sort(distances, order="distance")
                        neighbours = distances[: self.k_neighbors]
                        ref_neigh = random_state.choice(neighbours, 1)[0]
                        X_new, y_new = self._create_new_sample(
                            sample,
                            ref_neigh[1],
                            [x[1] for x in neighbours],
                            unique_labels,
                            labels,
                            random_state,
                        )
                        append_X_synth(X_new)
                        append_y_synth(y_new)
            return np.concatenate((X_resampled, np.array(X_synth))), np.array(
                y_resampled.tolist() + y_synth
            )

    def _validate_estimator(self):
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == bool:
            self.categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any(
                [cat not in np.arange(self.n_features_) for cat in categorical_features]
            ):
                raise ValueError(
                    "Some of the categorical indices are out of range. Indices"
                    f" should be between 0 and {self.n_features_ - 1}"
                )
            self.categorical_features_ = categorical_features
        self.continuous_features_ = np.setdiff1d(
            np.arange(self.n_features_), self.categorical_features_
        )

    def _collect_unique_labels(self, y):
        """A support function that flattens the labelsets and return one set of unique
        labels
        """
        return np.unique(
            np.array(
                [
                    label
                    for label_set in y
                    for label in (
                        label_set if isinstance(label_set, list) else [label_set]
                    )
                ]
            )
        )

    def _create_new_sample(
        self,
        sample_id,
        ref_neigh_id,
        neighbour_ids,
        unique_labels,
        labels,
        random_state,
    ):
        sample = self.features[sample_id]
        synth_sample = np.copy(sample)
        ref_neigh = self.features[ref_neigh_id]
        sample_labels = labels[sample_id]

        for i in range(synth_sample.shape[0]):
            if i in self.continuous_features_:
                diff = ref_neigh[i] - sample[i]
                offset = diff * random_state.uniform(0, 1)
                synth_sample[i] = sample[i] + offset
            if i in self.categorical_features_:
                synth_sample[i] = self._get_most_frequent_value(
                    self.features[neighbour_ids, i]
                )
        X = synth_sample

        if sparse.issparse(labels):
            neighbours_labels = labels[neighbour_ids]
            possible_labels = neighbours_labels.sum(axis=0)
            y = np.zeros((1, len(unique_labels)))
            if self.sampling_strategy_ == "ranking":
                head_index = int((self.k_neighbors + 1) / 2)
                choosen_labels = possible_labels.nonzero()[1][:head_index]
                y[0, choosen_labels] = 1
            if self.sampling_strategy_ == "union":
                choosen_labels = possible_labels.nonzero()[0]
                y[choosen_labels] = 1
            if self.sampling_strategy_ == "intersection":
                choosen_labels = sparse.find(possible_labels == len(neighbours_labels))
                y[choosen_labels] = 1
            y = sparse.csr_matrix(y)

        else:
            neighbours_labels = []
            for ni in neighbour_ids:
                neighbours_labels.append(labels[ni].tolist())

            new_labels = []  # sample_labels.tolist()
            new_labels += [
                a
                for x in neighbours_labels
                for a in (x if isinstance(x, list) else [x])
            ]
            new_labels = list(set(new_labels))
            if self.sampling_strategy_ == "ranking":
                head_index = int((self.k_neighbors + 1) / 2)
                y = new_labels[:head_index]
            if self.sampling_strategy_ == "union":
                y = new_labels[:]
            if self.sampling_strategy_ == "intersection":
                y = list(set.intersection(*neighbours_labels))

        return X, y

    def _calc_distances(self, sample, min_bag, unique_labels, labels):
        def calc_dist(bag_sample):
            nominal_distance = sum(
                [
                    self._get_vdm(
                        self.features[sample, cat],
                        self.features[bag_sample, cat],
                        cat,
                        unique_labels,
                        labels,
                    )
                    for cat in self.categorical_features_
                ]
            )
            ordinal_distance = sum(
                [
                    self._get_euclidean_distance(
                        self.features[sample, num], self.features[bag_sample, num]
                    )
                    for num in self.continuous_features_
                ]
            )
            dist = sum([nominal_distance, ordinal_distance])
            return (dist, bag_sample)

        distances = [calc_dist(bag_sample) for bag_sample in min_bag]
        dtype = np.dtype([("distance", float), ("index", int)])
        return np.array(distances, dtype=dtype)

    def _get_euclidean_distance(self, first, second):
        euclidean_distance = np.linalg.norm(first - second)
        return euclidean_distance

    def _get_vdm(self, first, second, category, unique_labels, labels):
        """A support function to compute the Value Difference Metric(VDM) discribed in https://arxiv.org/pdf/cs/9701101.pdf"""
        if sparse.issparse(self.features):

            def f_sparse(c):
                N_ax = len(sparse.find(self.features[:, category] == first)[0])
                N_ay = len(sparse.find(self.features[:, category] == second)[0])
                c_instances = self._get_all_instances_of_label(c, labels)
                N_axc = len(
                    sparse.find(self.features[c_instances, category] == first)[0]
                )
                N_ayc = len(
                    sparse.find(self.features[c_instances, category] == second)[0]
                )
                p = np.square(np.abs((N_axc / N_ax) - (N_ayc / N_ay)))
                return p

            vdm = np.sum(np.array([f_sparse(c) for c in unique_labels]))
            return vdm

        category_rows = self.features[:, category]
        N_ax = len(np.where(category_rows == first))
        N_ay = len(np.where(category_rows == second))

        def f(c):
            class_instances = self._get_all_instances_of_label(c, labels)
            class_instance_rows = category_rows[class_instances]
            N_axc = len(np.where(class_instance_rows == first)[0])
            N_ayc = len(np.where(class_instance_rows == second)[0])
            p = abs((N_axc / N_ax) - (N_ayc / N_ay))
            return p

        vdm = np.array([f(c) for c in unique_labels]).sum()
        return vdm

    def _get_all_instances_of_label(self, label, labels):
        if sparse.issparse(labels):
            return labels[:, label].nonzero()[0]
        instance_ids = []
        append_instance_id = instance_ids.append
        for i, label_set in enumerate(labels):
            if label in label_set:
                append_instance_id(i)
        return np.array(instance_ids)

    def _get_mean_imbalance_ratio(self, unique_labels, labels):
        irlbl_num = self._get_imbalance_ratio_numerator(unique_labels, labels)
        ratio_sum = np.sum(
            np.array(
                list(
                    map(
                        self._get_imbalance_ratio_per_label,
                        unique_labels,
                        itertools.repeat(irlbl_num),
                        itertools.repeat(labels),
                    )
                )
            )
        )
        return ratio_sum / len(unique_labels)

    def _get_imbalance_ratio_numerator(self, unique_labels, labels):
        sum_array = np.array([self._sum_h(label, labels) for label in unique_labels])
        return sum_array.max()

    def _get_imbalance_ratio_per_label(self, label, irlbl_numerator, labels):
        return irlbl_numerator / self._sum_h(label, labels)

    def _sum_h(self, label, labels):
        if sparse.issparse(labels):
            return labels[:, label].count_nonzero()

        h_sum = 0

        def h(l, Y):
            if l in Y:
                return 1
            else:
                return 0

        for label_set in labels:
            h_sum += h(label, label_set)

        return h_sum

    def _get_label_frequencies(self, labels):
        """A support function to get the frequencies of labels"""
        frequency_map = np.array(np.unique(labels, return_counts=True)).T
        frequencies = np.array([x[1] for x in frequency_map])
        return frequencies

    def _get_most_frequent_value(self, values):
        """A support function to get most frequent value if a list of values"""
        uniques, indices = np.unique(values, return_inverse=True)
        return uniques[np.argmax(np.bincount(indices))]
