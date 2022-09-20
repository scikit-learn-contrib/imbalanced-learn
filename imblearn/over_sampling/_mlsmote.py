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
        If ``int``, number of nearest neighbors used to construct synthetic
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

    _required_parameters = ["categorical_features"]
    _sampling_strategies = ["intersection", "ranking", "union"]

    def __init__(
        self,
        categorical_features,
        *,
        sampling_strategy="ranking",
        random_state=None,
        k_neighbors=5,
    ):
        if sampling_strategy not in MLSMOTE._sampling_strategies:
            raise ValueError(
                "Sampling Strategy can only be one of: 'ranking', 'union' or "
                "'intersection'"
            )

        self.categorical_features = categorical_features
        self.sampling_strategy_ = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def _validate_estimator(self):
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == "bool":
            self.categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any(
                [cat not in np.arange(self.n_features_) for cat in categorical_features]
            ):
                raise ValueError(
                    "Some of the categorical indices are out of range. Indices"
                    f" should be between 0 and {self.n_features_}"
                )
            self.categorical_features_ = categorical_features
        self.continuous_features_ = np.setdiff1d(
            np.arange(self.n_features_), self.categorical_features_
        )

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

        # Convert 'y' to a sparse matrix
        if type(y) == sparse._csr.csr_matrix:
            y_resampled = y.copy()
            unique_labels = range(0, y_resampled.shape[1])
        elif type(y) == np.ndarray:
            y_resampled = sparse.csr_matrix(y, dtype=int)
            unique_labels = range(0, y_resampled.shape[1])
        elif type(y) == list:
            unique_labels = self._collect_unique_labels(y)
            y_resampled = sparse.csr_matrix((len(y), len(unique_labels)))
            for i, sample in enumerate(y):
                for label in sample:
                    y_resampled[i, label] = 1
        else:
            raise TypeError(
                "'y' can only be of type 'numpy.ndarray', "
                "'scipy.sparse._csr.csr_matrix' or 'list'"
            )

        """TODO: Handle the case where 'mean_ir' is infinity. Happens when one label has
        no samples
        """
        mean_ir = self._get_mean_imbalance_ratio(unique_labels, y_resampled)

        for label in unique_labels:
            irlbl_num = self._get_imbalance_ratio_numerator(unique_labels, y_resampled)
            irlbl = self._get_imbalance_ratio_per_label(label, irlbl_num, y_resampled)
            if irlbl > mean_ir:
                min_bag = self._get_all_instances_of_label(label, y_resampled)
                if (
                    len(min_bag) <= 1
                ):  # If there is only one sample, the neighbor set will be empty
                    continue
                for sample_id in min_bag:
                    distances = self._calc_distances(
                        sample_id, min_bag, X_resampled, unique_labels, y_resampled
                    )
                    distances = np.sort(distances, order="distance")
                    neighbors = distances[
                        1 : self.k_neighbors + 1
                    ]  # Remove 'sample' from neighbor set
                    ref_neigh = random_state.choice(neighbors, 1)[0]
                    X_new, y_new = self._create_new_sample(
                        sample_id,
                        ref_neigh[1],
                        [x[1] for x in neighbors],
                        X_resampled,
                        unique_labels,
                        y_resampled,
                        random_state,
                    )
                    X_resampled = np.vstack((X_resampled, X_new))
                    y_resampled = sparse.vstack((y_resampled, y_new))
        return X_resampled, y_resampled

    def _create_new_sample(
        self,
        sample_id,
        ref_neigh_id,
        neighbor_ids,
        X_resampled,
        unique_labels,
        y_resampled,
        random_state,
    ):
        sample = X_resampled[sample_id]
        synth_sample = np.zeros_like(sample)
        ref_neigh = X_resampled[ref_neigh_id]

        for i in range(synth_sample.shape[0]):
            if i in self.continuous_features_:
                diff = ref_neigh[i] - sample[i]
                offset = diff * random_state.uniform(0, 1)
                synth_sample[i] = sample[i] + offset
            elif i in self.categorical_features_:
                synth_sample[i] = self._get_most_frequent_value(
                    X_resampled[neighbor_ids, i]
                )

        neighbors_labels = y_resampled[neighbor_ids]
        possible_labels = neighbors_labels.sum(axis=0)
        y = np.zeros((1, len(unique_labels)))
        if self.sampling_strategy_ == "ranking":
            head_index = int((self.k_neighbors + 1) / 2)
            choosen_labels = possible_labels.nonzero()[1][:head_index]
            y[0, choosen_labels] = 1
        elif self.sampling_strategy_ == "union":
            choosen_labels = possible_labels.nonzero()[0]
            y[choosen_labels] = 1
        elif self.sampling_strategy_ == "intersection":
            choosen_labels = sparse.find(possible_labels == len(neighbors_labels))
            y[choosen_labels] = 1
        y = sparse.csr_matrix(y)

        return synth_sample, y

    def _collect_unique_labels(self, y):
        """A support function that flattens the labelsets and return one set of unique
        labels
        """
        return np.unique(np.array([label for label_set in y for label in label_set]))

    def _calc_distances(self, sample, min_bag, features, unique_labels, labels):
        def calc_dist(bag_sample):
            nominal_distance = sum(
                [
                    self._get_vdm(
                        features[sample, cat],
                        features[bag_sample, cat],
                        features,
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
                        features[sample, num], features[bag_sample, num]
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

    def _get_vdm(self, first, second, features, category, unique_labels, labels):
        """A support function to compute the Value Difference Metric(VDM) discribed in
        https://arxiv.org/pdf/cs/9701101.pdf
        """
        if type(labels) == np.ndarray or type(labels) == sparse._csr.csr_matrix:

            def f_sparse(c):
                N_ax = len(sparse.find(features[:, category] == first)[0])
                N_ay = len(sparse.find(features[:, category] == second)[0])
                c_instances = self._get_all_instances_of_label(c, labels)
                N_axc = len(sparse.find(features[c_instances, category] == first)[0])
                N_ayc = len(sparse.find(features[c_instances, category] == second)[0])
                p = np.square(np.abs((N_axc / N_ax) - (N_ayc / N_ay)))
                return p

            vdm = np.sum(np.array([f_sparse(c) for c in unique_labels]))
            return vdm

        category_rows = features[:, category]
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
        return labels[:, label].nonzero()[0]

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
        return labels[:, label].count_nonzero()

    def _get_label_frequencies(self, labels):
        """A support function to get the frequencies of labels"""
        frequency_map = np.array(np.unique(labels, return_counts=True)).T
        frequencies = np.array([x[1] for x in frequency_map])
        return frequencies

    def _get_most_frequent_value(self, values):
        """A support function to get most frequent value if a list of values
        TODO: We might want to randomize 'unique' and 'counts' to avoid always returning
        the first occurrence when multiple occurrences of the maximum value.
        """
        uniques, counts = np.unique(values, return_counts=True)
        return uniques[np.argmax(counts)]
