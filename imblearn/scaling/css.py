"""Class to perform sample scaling using class specific scaling (CSS)."""
# Authors: Bernhard Schlegel <bernhard.schlegel@mytum.de>
# License: MIT


from __future__ import division, print_function
from collections import Counter
import random
import numpy as np
from sklearn.utils import check_random_state, safe_indexing

from .base import BaseScaler

CSS_MODE = ('linear', 'constant')
CSS_SAMPLING_STRATEGY = ('minority', 'majority', 'both')


class CSS(BaseScaler):
    """Class to perform sample scaling using class specific scaling (CSS).

    Parameters
    ----------
    mode : str (default = 'constant')
        Defines the scaling mode. Currently, two modes are implemented: `'constant'`
        and `'linear'`. 
        
        In `'constant'` mode, all samples of the `'sampling_strategy'` class will be scaled
        by the same amount `c` to their class specific center. The following 
        formula will be applied to calculate the new feature (`X`) values:
        `X[y==0] * (1-c) + col_means * c`
        
        In `'linear'` mode, all samples will be scaled in depedence on their 
        distance and `c` to their class specific center. Samples, that are 
        one/unit standard deviation away from the class center will be scaled 
        with `c`. The following formula will be applied to calculate the new 
        feature (`X`) values:
        `norm = distances * c + (1-c)`
        `X[y==0] * (1-c) / norm + col_means * (distances * c) / norm
        

    sampling_strategy : str (default = 'minority')
        defines which class to scale. Possible values are 'minority', 'majority',
        and 'both'. Note that all sample are scaled to their corresponding class
        center.

    c : float (default = 0.25)
        Defines the amount of the scaling. 
    
    sampling_strategy_class_value: int (default = None)
        class level indicating the minority class. By default (`None`) the minority
        class will be automatically determined. Use any integer number (e.g. `0`,
        `1` or `-1`) to force the minority class.
        
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    Attributes
    ----------
    mode_ : str
        CSS mode ('constant' or 'linear')

    sampling_strategy_ : str or int
        Name of the sampling_strategy class ('majority', 'minority', 'both')
         
    sampling_strategy_class_value: int
        class level indicating the minority class

    c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.
        
    shuffle : Boolean
        If True, results will be shuffled.

    Examples
    --------

	>>> import numpy as np
	>>> from sklearn.utils import shuffle
	>>> from imblearn.scaling import CSS
	>>> rng = np.random.RandomState(42)
	>>> n_samples_1 = 50
	>>> n_samples_2 = 5
	>>> X_syn = np.r_[1.5 * rng.randn(n_samples_1, 2), 0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
	>>> y_syn = np.array([0] * (n_samples_1) + [1] * (n_samples_2))
	>>> X_syn, y_syn = shuffle(X_syn, y_syn)
	>>> css = CSS(mode="linear", sampling_strategy="both", c=0.1, shuffle=True)
	>>> X_train_res, y_train_res = css.fit_sample(X_syn, y_syn)

    References
    ----------
    .. [1] B. Schlegel, and B. Sick. "Dealing with class imbalance the scalable way: 
	       Evaluation of various techniques based on classification grade and computational 
		   complexity." 2017 IEEE International Conference on Data Mining Workshops, 2017.
    """

    def __init__(self,
                 sampling_strategy='minority',
                 mode='linear',
                 c=0.25,
                 minority_class_value=None,
                 shuffle=True):
        super(CSS, self).__init__()
        self.sampling_strategy = sampling_strategy
        self.mode = mode
        self.c = c
        self.minority_class_value = minority_class_value
        self.shuffle = shuffle

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be scaled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """

        super(CSS, self).fit(X, y)

        return self

    def _shuffleTwo(self, a, b):

        indexes = np.array(range(0, len(a)))
        random.shuffle(indexes)
        a2, b2 = a[indexes], b[indexes]

        return a2, b2, indexes

    def _sample(self, X, y):
        """scales the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_scaled : ndarray, shape (n_samples, n_features)
            The array containing the resampled data.

        y_scaled : ndarray, shape (n_samples)
            The corresponding label of `X_scaled`

        """

        if self.mode not in CSS_MODE:
            raise ValueError('Unknown kind for CSS mode.'
                             ' Choices are {}. Got \'{}\' instead.'.format(
                            CSS_MODE, self.mode))

        if self.sampling_strategy not in CSS_SAMPLING_STRATEGY:
            raise ValueError('Unknown kind for CSS sampling_strategy.'
                             ' Choices are {}. Got \'{}\' instead.'.format(
                                CSS_SAMPLING_STRATEGY, self.sampling_strategy))

        if self.c < 0 or self.c > 1:
            raise ValueError('Received scaling factor c={}, which'
                             ' is outside the allowed range '
                             '(0-1].'.format(self.c))
        if self.c is 0:
            raise ValueError('Received scaling factor c={}, which is'
                             ' equal to no CSS at.'.format(self.c))

        if self.minority_class_value is not None and \
                not isinstance(self.minority_class_value, int):
            raise ValueError('Unallowed sampling_strategy class value \'{}\'.'
                             ' Valid values include None to automatically'
                             ' infer the sampling_strategy class or any integer number'
                             ' corresponding to the value of the label in y')

        minority_class = self.minority_class_value
        if minority_class is None:
            # infer minority class value
            counts = Counter(y)
            least_common = counts.most_common()[:-1-1:-1]
            minority_class = least_common[0][0]

        # get indices for later, safe indexing
        majority_class_indices = (y != minority_class)
        minority_class_indices = (y == minority_class)

        # in the following _majority is majority, _minority is minority
        if self.sampling_strategy is "majority" or self.sampling_strategy is "both":
            # mean_majority_class is the mean of all features (=columns)
            mean_majority_class = np.mean(safe_indexing(X, majority_class_indices), axis=0)
            if self.mode is "linear":
                distances_majority = abs(np.subtract(safe_indexing(X, majority_class_indices), mean_majority_class))
        if self.sampling_strategy is "minority" or self.sampling_strategy is "both":
            mean_minority_class = np.mean(safe_indexing(X, minority_class_indices), axis=0)
            if self.mode is "linear":
                distances_minority = abs(np.subtract(safe_indexing(X, minority_class_indices), mean_minority_class))

        if self.sampling_strategy is "majority" or self.sampling_strategy is "both":
            if self.mode is "constant":
                X_scaled_majority = safe_indexing(X, majority_class_indices) * (1 - self.c) + mean_majority_class * self.c
            elif self.mode is "linear":
                scale_factors_mean = (distances_majority * self.c)
                scale_factors_values = (1 - self.c * distances_majority)

                X_scaled_majority = safe_indexing(X, majority_class_indices) * scale_factors_values + mean_majority_class * scale_factors_mean
        if self.sampling_strategy is "minority" or self.sampling_strategy is "both":
            if self.mode is "constant":
                X_scaled_minority = safe_indexing(X, minority_class_indices) * (1 - self.c) + mean_minority_class * self.c
            elif self.mode is "linear":
                scale_factors_mean = (distances_minority * self.c)
                scale_factors_values = (1 - self.c * distances_minority)
                X_scaled_minority = safe_indexing(X, minority_class_indices) * scale_factors_values + mean_minority_class * scale_factors_mean

        # merge scaled and non scaled stuff
        if self.sampling_strategy is "majority":
            X_scaled = np.concatenate([X_scaled_majority, safe_indexing(X, minority_class_indices)], axis=0)
        elif self.sampling_strategy is "minority":
            X_scaled = np.concatenate([safe_indexing(X, majority_class_indices), X_scaled_minority], axis=0)
        else: #"both"
            X_scaled = np.concatenate([X_scaled_majority, X_scaled_minority], axis=0)

        # make sure that y is in same order like X
        y_assembled = np.concatenate([y[majority_class_indices], y[minority_class_indices]], axis=0)

        # shuffle
        X_scaled_shuffled, y_res_shuffled, indices = self._shuffleTwo(X_scaled, y_assembled)

        if self.shuffle:
            return X_scaled_shuffled, y_res_shuffled
        else:
            return X_scaled, y_assembled
