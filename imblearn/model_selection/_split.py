import warnings

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples


class InstanceHardnessCV(BaseCrossValidator):
    """Instance-hardness cross-validation splitter.

    Cross-validation splitter that distributes samples with large instance hardness
    equally over the folds. The instance hardness is internally estimated by using
    `estimator` and stratified cross-validation.

    Read more in the :ref:`User Guide <instance_hardness_threshold_cv>`.

    Parameters
    ----------
    estimator : estimator object
        Classifier to be used to estimate instance hardness of the samples.
        This classifier should implement `predict_proba`.

    n_splits : int, default=5
        Number of folds. Must be at least 2.

    pos_label : int, float, bool or str, default=None
        The class considered the positive class when selecting the probability
        representing the instance hardness. If None, the positive class is
        automatically inferred by the estimator as `estimator.classes_[1]`.

    Examples
    --------
    >>> from imblearn.model_selection import InstanceHardnessCV
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(weights=[0.9, 0.1], class_sep=2,
    ... n_informative=3, n_redundant=1, flip_y=0.05, n_samples=1000, random_state=10)
    >>> estimator = LogisticRegression()
    >>> ih_cv = InstanceHardnessCV(estimator)
    >>> cv_result = cross_validate(estimator, X, y, cv=ih_cv)
    >>> print(f"Standard deviation of test_scores: {cv_result['test_score'].std():.3f}")
    Standard deviation of test_scores: 0.00...
    """

    def __init__(self, estimator, *, n_splits=5, pos_label=None):
        self.estimator = estimator
        self.n_splits = n_splits
        self.pos_label = pos_label

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )

        classes = np.unique(y)
        y_type = type_of_target(y)
        if y_type != "binary":
            raise ValueError("InstanceHardnessCV only supports binary classification.")
        if self.pos_label is None:
            pos_label = 1
        else:
            pos_label = np.flatnonzero(classes == self.pos_label)[0]

        y_proba = cross_val_predict(
            clone(self.estimator), X, y, cv=self.n_splits, method="predict_proba"
        )
        # sorting first on y and then by the instance hardness
        sorted_indices = np.lexsort((y_proba[:, pos_label], y))
        groups = np.empty(_num_samples(X), dtype=int)
        groups[sorted_indices] = np.arange(_num_samples(X)) % self.n_splits
        cv = LeaveOneGroupOut()
        for train_index, test_index in cv.split(X, y, groups):
            yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X: object
            Always ignored, exists for compatibility.

        y: object
            Always ignored, exists for compatibility.

        groups: object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits: int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits
