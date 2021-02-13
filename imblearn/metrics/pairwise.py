"""Metrics to perform pairwise computation."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np
from scipy.spatial import distance_matrix


class ValueDifferenceMetric:
    r"""Class implementing the Value Difference Metric.

    This metric computes the distance between samples containing only nominal
    categorical features. The distance between feature values of two samples
    is defined as:

    .. math::
       \delta(x, y) = \sum_{c=1}^{C} |p(c|x_{f}) - p(c|y_{f})|^{k} \ ,

    where :math:`x` and :math:`y` are two samples and :math:`f` a given
    feature, :math:`C` is the number of classes, :math:`p(c|x_{f})` is the
    conditional probability that the output class is :math:`c` given that
    the feature value :math:`f` has the value :math:`x` and :math:`k` an
    exponent usually defined to 1 or 2.

    The distance for the feature vectors :math:`X` and :math:`Y` is
    subsequently defined as:

    .. math::
       \Delta(X, Y) = \sum_{f=1}^{F} \delta(X_{f}, Y_{f})^{r} \ ,

    where :math:`F` is the number of feature and :math:`r` an exponent usually
    defined equal to 1 or 2.

    The definition of this distance was propoed in [1]_.

    Parameters
    ----------
    classes : ndarray of shape (n_classes,)
        The unique labels in `y`.

    categories : list of arrays
        List of arrays containing the categories for each feature. You can pass
        the fitted attribute `categories_` of the
        :class:`~sklearn.preprocesssing.OrdinalEncoder` used to encode the
        data.

    k : int, default=1
        Exponent used to compute the distance between feature value.

    r : int, default=2
        Exponent used to compute the distance between the feature vector.

    Attributes
    ----------
    proba_per_class_ : list of ndarray of shape (n_categories, n_classes)
        List of length `n_features` containing the conditional probabilities
        for each category given a class.

    References
    ----------
    .. [1] Stanfill, Craig, and David Waltz. "Toward memory-based reasoning."
       Communications of the ACM 29.12 (1986): 1213-1228.
    """

    def __init__(self, classes, categories, *, k=1, r=2):
        self.classes = classes
        self.categories = categories
        self.k = k
        self.r = r

    def fit(self, X, y):
        """Compute the necessary statistics from the training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), dtype={np.int32, np.int64}
            The input data. The data are expected to be encoded with an
            :class:`~sklearn.preprocessing.OrdinalEncoder`.

        y : ndarray of shape (n_features,)
            The target.

        Returns
        -------
        self
        """
        X = np.array(X, dtype=np.int32, copy=False)
        n_features = X.shape[1]

        # list of length n_features of ndarray (n_categories, n_classes)
        # compute the counts
        self.proba_per_class_ = [
            np.array(
                [
                    np.bincount(
                        X[y == klass, feature_idx],
                        minlength=len(self.categories[feature_idx]),
                    )
                    for klass in self.classes
                ],
                dtype=np.float64,
            ).T
            for feature_idx in range(n_features)
        ]
        # normalize by the summing over the classes
        for feature_idx in range(n_features):
            self.proba_per_class_[feature_idx] /= (
                self.proba_per_class_[feature_idx].sum(axis=1).reshape(-1, 1)
            )

        return self

    def pairwise(self, X1, X2=None):
        """Compute the VDM distance pairwise.

        Parameters
        ----------
        X1 : ndarray of shape (n_samples, n_features), dtype={np.int32, np.int64}
            The input data. The data are expected to be encoded with an
            :class:`~sklearn.preprocessing.OrdinalEncoder`.

        X2 : ndarray of shape (n_samples, n_features), dtype={np.int32, np.int64}
            The input data. The data are expected to be encoded with an
            :class:`~sklearn.preprocessing.OrdinalEncoder`.

        Returns
        -------
        distance_matrix : ndarray of shape (n_samples, n_samples)
            The VDM pairwise distance.
        """
        if X1.dtype.kind != "i":
            X1 = X1.astype(np.int32)
        n_samples_X1, n_features = X1.shape

        if X2 is not None:
            if X2.dtype.kind != "i":
                X2 = X2.astype(np.int32)
            n_samples_X2 = X2.shape[0]
        else:
            n_samples_X2 = n_samples_X1

        distance = np.zeros(shape=(n_samples_X1, n_samples_X2), dtype=np.float64)
        for feature_idx in range(n_features):
            proba_feature_X1 = self.proba_per_class_[feature_idx][X1[:, feature_idx]]
            if X2 is not None:
                proba_feature_X2 = self.proba_per_class_[feature_idx][
                    X2[:, feature_idx]
                ]
            else:
                proba_feature_X2 = proba_feature_X1
            distance += (
                distance_matrix(proba_feature_X1, proba_feature_X2, p=self.k) ** self.r
            )
        return distance
