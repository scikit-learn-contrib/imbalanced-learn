"""Class to perfrom over-sampling using MLSMOTE."""

from itertools import combinations
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
    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from imblearn.over_sampling import MLSMOTE
    >>> X, y = make_multilabel_classification(n_classes=5, n_features=20,
    ... random_state=42)
    >>> print("Original Dataset")
    Original Dataset
    >>> print(f"Samples: {X.shape[0]}")
    Samples: 100
    >>> for _class in range(y.shape[1]):
    ...     print(f"Class {_class} count: {np.count_nonzero(y[:, _class])}")
    Class 0 count: 30
    Class 1 count: 54
    Class 2 count: 48
    Class 3 count: 33
    Class 4 count: 14
    >>> categorical_features = np.full((20,), True)
    >>> mlsmote = MLSMOTE(categorical_features, random_state=42)
    >>> X_res, y_res = mlsmote.fit_resample(X, y)
    >>> print("Resampled Dataset")
    Resampled Dataset
    >>> print(f"Samples: {X_res.shape[0]}")
    Samples: 114
    >>> for _class in range(y_res.shape[1]):
    ...     print(f"Class {_class} count: {np.count_nonzero(y_res[:, _class])}")
    Class 0 count: 30
    Class 1 count: 60
    Class 2 count: 56
    Class 3 count: 33
    Class 4 count: 28
    """

    _required_parameters = ["categorical_features"]

    INTERSECTION = "intersection"
    RANKING = "ranking"
    UNION = "union"
    _sampling_strategies = [INTERSECTION, RANKING, UNION]

    def __init__(
        self,
        categorical_features,
        *,
        sampling_strategy=RANKING,
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
                (n_samples, n_labels) or a list of lists of labels.
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

        y_resampled : array-like of shape (n_samples_new, n_labels) \
                or a list of lists of labels.
            The corresponding label sets of `X_resampled`.
        """
        self.n_features_ = X.shape[1]

        self._validate_estimator()
        random_state = check_random_state(self.random_state)

        X_resampled = X.copy()

        unique_labels = None
        # Convert 'y' to a sparse matrix
        if type(y) == sparse._csr.csr_matrix:
            y_resampled = y.copy()
        elif type(y) == np.ndarray:
            y_resampled = sparse.csr_matrix(y, dtype=int)
        elif type(y) == list:
            unique_labels = self._collect_unique_labels(y)
            y_resampled = sparse.csr_matrix((len(y), len(unique_labels)))
            for i, sample_labels in enumerate(y):
                for label in sample_labels:
                    y_resampled[i, np.where(unique_labels == label)] = 1
        else:
            raise TypeError(
                "'y' can only be of type 'numpy.ndarray', "
                "'scipy.sparse._csr.csr_matrix' or 'list'"
            )

        self.n_classes_ = y_resampled.shape[1]

        """TODO: Handle the case where 'mean_ir' is infinity. Happens when one label has
        no samples
        """
        mean_ir = self._get_mean_imbalance_ratio(y_resampled)

        for label in range(self.n_classes_):
            irlbl_num = self._get_imbalance_ratio_numerator(y_resampled)
            irlbl = self._get_imbalance_ratio_per_label(label, irlbl_num, y_resampled)
            if irlbl > mean_ir:
                min_bag = self._get_all_instances_of_label(label, y_resampled)
                euclidean_dist_cache = np.zeros((y_resampled.shape[0], y_resampled.shape[0]))
                X_sliced = X_resampled[:][:,self.continuous_features_]
                pairs = list(combinations(min_bag, 2))
                for m, n in pairs:
                    distance = sum(self._get_euclidean_distance(
                        X_sliced[m, :], X_sliced[n, :]
                    ))
                    euclidean_dist_cache[m, n] = distance
                    euclidean_dist_cache[n, m] = distance
                if (
                    len(min_bag) <= 1
                ):  # If there is only one sample, the neighbor set will be empty
                    continue
                for sample_id in min_bag:
                    distances = self._calc_distances(
                        sample_id, min_bag, X_resampled, y_resampled, euclidean_dist_cache,
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
                        y_resampled,
                        random_state,
                    )
                    X_resampled = np.vstack((X_resampled, X_new))
                    y_resampled = sparse.vstack((y_resampled, y_new))
        return X_resampled, self._convert_to_input_type(
            y_resampled, unique_labels, type(y)
        )

    def _create_new_sample(
        self,
        sample_id,
        ref_neigh_id,
        neighbor_ids,
        X_resampled,
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
        label_counts = np.squeeze(
            np.asarray(y_resampled[sample_id] + neighbors_labels.sum(axis=0))
        )
        synth_sample_labels = sparse.csr_matrix((1, self.n_classes_), dtype=int)
        if self.sampling_strategy_ == MLSMOTE.RANKING:
            # Note: Paper states "present in half or more of the instances considered"
            # but pseudocode shows: "labels lblCounts > (k + 1)/2" instead of '>='. We
            # follow the pseudocode for now.
            quorum = int((len(neighbor_ids) + 1) / 2)
            chosen_labels = label_counts > quorum
        elif self.sampling_strategy_ == MLSMOTE.UNION:
            chosen_labels = label_counts.nonzero()
        elif self.sampling_strategy_ == MLSMOTE.INTERSECTION:
            chosen_labels = label_counts == len(neighbor_ids) + 1

        synth_sample_labels[0, chosen_labels] = 1

        return synth_sample, synth_sample_labels

    def _collect_unique_labels(self, y):
        """A support function that flattens the labelsets and return one set of unique
        labels
        """
        return np.unique(np.array([label for label_set in y for label in label_set]))

    def _calc_distances(self, sample, min_bag, features, labels, euclidean_dist_cache):
        def calc_dist(bag_sample):
            nominal_distance = sum(
                [
                    self._get_vdm(
                        features[sample, cat],
                        features[bag_sample, cat],
                        features,
                        cat,
                        labels,
                    )
                    for cat in self.categorical_features_
                ]
            )
            ordinal_distance = euclidean_dist_cache[sample, bag_sample]
            dist = nominal_distance + ordinal_distance
            return (dist, bag_sample)

        distances = [calc_dist(bag_sample) for bag_sample in min_bag]
        dtype = np.dtype([("distance", float), ("index", int)])
        return np.array(distances, dtype=dtype)

    def _get_euclidean_distance(self, first, second):
        """Since the inputs are of type 'float' the euclidean distance is just
        the absolute value of their difference.
        """
        return abs(first - second)

    def _get_vdm(self, x_attr_val, y_attr_val, features, category, labels):
        """A support function to compute the Value Difference Metric(VDM) described in
        https://arxiv.org/pdf/cs/9701101.pdf
        """

        def f_sparse(_class):
            c_instances = self._get_all_instances_of_label(_class, labels)
            N_axc = np.count_nonzero(features[c_instances, category] == x_attr_val)
            N_ayc = np.count_nonzero(features[c_instances, category] == y_attr_val)
            p = abs((N_axc / N_ax) - (N_ayc / N_ay)) ** 2
            return p

        N_ax = np.count_nonzero(features[:, category] == x_attr_val)
        N_ay = np.count_nonzero(features[:, category] == y_attr_val)
        vdm = sum([f_sparse(_class) for _class in range(self.n_classes_)])
        return vdm

    def _get_all_instances_of_label(self, label, labels):
        return labels[:, label].nonzero()[0]

    def _get_mean_imbalance_ratio(self, labels):
        sum_per_label = np.array(
            [self._sum_h(label, labels) for label in range(self.n_classes_)]
        )
        irlbl_num = sum_per_label.max()
        ratio_sum = np.sum(irlbl_num / sum_per_label)
        return ratio_sum / self.n_classes_

    def _get_imbalance_ratio_numerator(self, labels):
        sum_array = np.array(
            [self._sum_h(label, labels) for label in range(self.n_classes_)]
        )
        return sum_array.max()

    def _get_imbalance_ratio_per_label(self, label, irlbl_numerator, labels):
        return irlbl_numerator / self._sum_h(label, labels)

    def _sum_h(self, label, labels):
        return labels[:, label].count_nonzero()

    def _get_most_frequent_value(self, values):
        """A support function to get most frequent value if a list of values
        TODO: We might want to randomize 'unique' and 'counts' to avoid always returning
        the first occurrence when multiple occurrences of the maximum value.
        """
        uniques, counts = np.unique(values, return_counts=True)
        return uniques[np.argmax(counts)]

    def _convert_to_input_type(self, y_resampled, unique_labels, input_type):
        """A support function that converts the labels back to its input format"""
        if input_type == sparse._csr.csr_matrix:
            return y_resampled
        elif input_type == np.ndarray:
            return np.asarray(y_resampled.todense())
        elif input_type == list:
            labels = [[] for _ in range(y_resampled.shape[0])]
            rows, cols = y_resampled.nonzero()
            for row, col in zip(rows, cols):
                labels[row].append(unique_labels[col])
            return labels
