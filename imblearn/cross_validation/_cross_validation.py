import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict


class InstanceHardnessCV:
    """Instance-hardness CV splitter

    CV splitter that distributes samples with large instance hardness equally
    over the folds

    Read more in the :ref:`User Guide <instance_hardness_threshold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    clf : classifier, default=None
        Classifier used to determine instance hardness. Defaults to
        RandomForestClassifier when set to `None`

    random_state : int, RandomState instance, default=None
        Determines random_state for reproducible results across multiple calls.

    Examples
    --------
    >>> from imblearn.cross_validation import InstanceHardnessCV
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(weights=[0.9, 0.1], class_sep=2,
    ... n_informative=3, n_redundant=1, flip_y=0.05, n_samples=1000, random_state=10)
    >>> ih_cv = InstanceHardnessCV(n_splits=5, random_state=10)
    >>> clf = LogisticRegression(random_state=10)
    >>> cv_result = cross_validate(clf, X, y, cv=ih_cv)
    >>> print(f"Standard deviation of test_scores: {cv_result['test_score'].std():.3f}")
    Standard deviation of test_scores: 0.005
    """

    def __init__(self, n_splits=5, clf=None, random_state=None):
        self.n_splits = n_splits
        self.clf = clf
        self.random_state = random_state

    def split(self, X, y, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        y: array-like of shape (n_samples,)
            The target variable.

        groups: object
            Always ignored, exists for compatibility.

        Yields
        ------

        train: ndarray
            The training set indices for that split.

        test: ndarray
            The testing set indices for that split.

        """
        df = pd.DataFrame(X)
        features = df.columns
        df["y"] = y
        if self.clf is not None:
            self.clf_ = self.clf
        else:
            self.clf_ = RandomForestClassifier(
                n_jobs=-1, class_weight="balanced", random_state=self.random_state
            )
        df["proba"] = cross_val_predict(
            self.clf_, df[features], df["y"], cv=self.n_splits, method="predict_proba"
        )[:, 1]
        df["hardness"] = abs(df["y"] - df["proba"])
        df = df.sort_values("hardness")
        df["group"] = np.arange(len(df)) % self.n_splits
        cv = StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for train_index, test_index in cv.split(df[features], df["y"], df["group"]):
            yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

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
