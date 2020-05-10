import numpy as np
import itertools
import collections
import random


class MLSMOTE:
    """Over-sampling using MLSMOTE.

    Parameters
    ----------
    sampling_strategy: 'ranking','union' or 'intersection' default: 'ranking'
        Strategy to generate labelsets


    k_neighbors : int or object, default=5
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.

    categorical_features : ndarray of shape (n_cat_features,) or (n_features,)
        Specified which features are categorical. Can either be:

        - array of indices specifying the categorical features;
        - mask array of shape (n_features, ) and ``bool`` dtype for which
          ``True`` indicates the categorical features.

    Notes
    -----
    See the original papers: [1]_ for more details.


    References
    ----------
    .. [1]  Charte, F. & Rivera Rivas, Antonio & Del Jesus, María José & Herrera, Francisco. (2015).
            MLSMOTE: Approaching imbalanced multilabel learning through synthetic instance generation.
            Knowledge-Based Systems. -. 10.1016/j.knosys.2015.07.019. 

    """

    def __init__(self, categorical_features, k_neighbors=5, sampling_strategy='ranking'):
        self.k_neighbors = k_neighbors
        self.sampling_strategy_ = sampling_strategy
        self.categorical_features = categorical_features
        self.continuous_features_ = None
        self.unique_labels = []
        self.labels = []
        self.features = []

    def fit_resample(self, X, y):
        self.n_features_ = X.shape[1]
        self.labels = np.array([np.array(xi) for xi in y])

        self._validate_estimator()

        X_resampled = X.copy()
        y_resampled = y.copy()

        self.unique_labels = self._collect_unique_labels(y)
        self.features = X

        X_synth = []
        y_synth = []

        append_X_synth = X_synth.append
        append_y_synth = y_synth.append
        mean_ir = self._get_mean_imbalance_ratio()
        for label in self.unique_labels:
            irlbl = self._get_imbalance_ratio_per_label(label)
            if irlbl > mean_ir:
                min_bag = self._get_all_instances_of_label(label)
                for sample in min_bag:
                    distances = self._calc_distances(sample, min_bag)
                    distances = np.sort(distances, order='distance')
                    neighbours = distances[:self.k_neighbors]
                    ref_neigh = np.random.choice(neighbours, 1)[0]
                    X_new, y_new = self._create_new_sample(
                        sample, ref_neigh[1], [x[1] for x in neighbours])
                    append_X_synth(X_new)
                    append_y_synth(y_new)

        return np.concatenate((X_resampled, np.array(X_synth))), np.array(y_resampled.tolist()+y_synth)

    def _validate_estimator(self):
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == "bool":
            self.categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any(
                [
                    cat not in np.arange(self.n_features_)
                    for cat in categorical_features
                ]
            ):
                raise ValueError(
                    "Some of the categorical indices are out of range. Indices"
                    " should be between 0 and {}".format(self.n_features_)
                )
            self.categorical_features_ = categorical_features
        self.continuous_features_ = np.setdiff1d(
            np.arange(self.n_features_), self.categorical_features_
        )

    def _collect_unique_labels(self, y):
        """A support function that flattens the labelsets and return one set of unique labels"""
        return np.unique(np.array([a for x in y for a in (x if isinstance(x, list) else [x])]))

    def _create_new_sample(self, sample_id, ref_neigh_id, neighbour_ids):
        sample = self.features[sample_id]
        sample_labels = self.labels[sample_id]
        synth_sample = np.copy(sample)
        ref_neigh = self.features[ref_neigh_id]
        neighbours_labels = []
        for ni in neighbour_ids:
            neighbours_labels.append(self.labels[ni].tolist())
        for i in range(synth_sample.shape[0]):
            if i in self.continuous_features_:
                diff = ref_neigh[i]-sample[i]
                offset = diff*random.uniform(0, 1)
                synth_sample[i] = sample[i]+offset
            if i in self.categorical_features_:
                synth_sample[i] = self._get_most_frequent_value(
                    self.features[neighbour_ids, i])

        labels = sample_labels.tolist()
        labels += [a for x in neighbours_labels for a in (
            x if isinstance(x, list) else [x])]
        labels = list(set(labels))
        if self.sampling_strategy_ == 'ranking':
            head_index = int((self.k_neighbors + 1)/2)
            y = labels[:head_index]
        if self.sampling_strategy_ == 'union':
            y = labels[:]
        if self.sampling_strategy_ == 'intersection':
            y = list(set.intersection(*neighbours_labels))

        X = synth_sample
        return X, y

    def _calc_distances(self, sample, min_bag):
        distances = []
        append_distances = distances.append
        for bag_sample in min_bag:
            nominal_distances = np.array([self._get_vdm(
                self.features[sample, cat], self.features[bag_sample, cat])for cat in self.categorical_features_])
            ordinal_distances = np.array([self._get_euclidean_distance(
                self.features[sample, num], self.features[bag_sample, num])for num in self.continuous_features_])
            dists = np.array(
                [nominal_distances.sum(), ordinal_distances.sum()])
            append_distances((dists.sum(), bag_sample))
        dtype = np.dtype([('distance', float), ('index', int)])
        return np.array(distances, dtype=dtype)

    def _get_euclidean_distance(self, first, second):
        euclidean_distance = np.linalg.norm(first-second)
        return euclidean_distance

    def _get_vdm(self, first, second):
        """A support function to compute the Value Difference Metric(VDM) discribed in https://arxiv.org/pdf/cs/9701101.pdf"""
        def f(c):
            N_ax = len(
                np.where(self.features[:, self.categorical_features_] == first))
            N_ay = len(
                np.where(self.features[:, self.categorical_features_] == second))
            c_instances = self._get_all_instances_of_label(c)
            N_axc = len(np.where(self.features[np.ix_(
                c_instances, self.categorical_features_)] == first)[0])
            N_ayc = len(np.where(self.features[np.ix_(
                c_instances, self.categorical_features_)] == second)[0])
            return np.square(np.abs((N_axc/N_ax)-(N_ayc/N_ay)))

        return np.sum(np.array([f(c)for c in self.unique_labels]))

    def _get_all_instances_of_label(self, label):
        instance_ids = []
        append_instance_id = instance_ids.append
        for i, label_set in enumerate(self.labels):
            if label in label_set:
                append_instance_id(i)
        return np.array(instance_ids)

    def _get_mean_imbalance_ratio(self):
        ratio_sum = np.sum(
            np.array(list(map(self._get_imbalance_ratio_per_label, self.unique_labels))))
        return ratio_sum/self.unique_labels.shape[0]

    def _get_imbalance_ratio_per_label(self, label):
        sum_array = list(map(self._sum_h, self.unique_labels))
        sum_array = np.array(sum_array)
        return sum_array.max()/self._sum_h(label)

    def _sum_h(self, label):
        h_sum = 0

        def h(l, Y):
            if l in Y:
                return 1
            else:
                return 0

        for label_set in self.labels:
            h_sum += h(label, label_set)
        return h_sum

    def _get_label_frequencies(self, labels):
        """"A support function to get the frequencies of labels"""
        frequency_map = np.array(np.unique(labels, return_counts=True)).T
        frequencies = np.array([x[1] for x in count_map])
        return frequencies

    def _get_most_frequent_value(self, values):
        """"A support function to get most frequent value if a list of values"""
        uniques, indices = np.unique(values, return_inverse=True)
        return uniques[np.argmax(np.bincount(indices))]
