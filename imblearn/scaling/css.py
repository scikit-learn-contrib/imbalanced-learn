"""Class to perform sample scaling using class specific scaling (CSS)."""
# Authors: Bernhard Schlegel <bernhard.schlegel@mytum.de>
# License: MIT


from __future__ import division, print_function
from collections import Counter
import random
import numpy as np
from .base import BaseScaler

CSS_MODE = ('linear', 'constant')
CSS_TARGET = ('minority', 'majority', 'both')


class CSS(BaseScaler):
    """Class to perform sample scaling using class specific scaling (CSS).

    Parameters
    ----------
    mode : str (default = 'constant')
        Defines the scaling mode. Currently, two modes are implemented: `'constant'`
        and `'linear'`. 
        
        In `'constant'` mode, all samples of the `'target'` class will be scaled
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
        

    target : str (default = 'minority')
        defines which class to scale. Possible values are 'minority', 'majority',
        and 'both'. Note that all sample are scaled to their corresponding class
        center.

    c : float (default = 0.25)
        Defines the amount of the scaling. 
    
    target_class_value: int (default = None)
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

    target_ : str or int
        Name of the target class ('majority', 'minority', 'both')
         
    target_class_value: int
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
	>>> X_syn = np.r_[1.5 * rng.randn(n_samples_1, 2),
	    			  0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
	>>> y_syn = np.array([0] * (n_samples_1) + [1] * (n_samples_2))
	>>> X_syn, y_syn = shuffle(X_syn, y_syn)
	>>> css = CSS(mode="linear", target="both", c=0.1, shuffle=True)
	>>> X_train_res, y_train_res = css.fit_sample(X_syn, y_syn)

    References
    ----------
    .. [1] B. Schlegel, and B. Sick. "Dealing with class imbalance the scalable way: 
	       Evaluation of various techniques based on classification grade and computational 
		   complexity." 2017 IEEE International Conference on Data Mining Workshops, 2017.
    """

    def __init__(self,
                 mode='linear',
                 target='minority',
                 c=0.25,
                 minority_class_value=None,
                 shuffle=True,
                 random_state=None):
        super(CSS, self).__init__(ratio=1)
        self.mode = mode
        self.target = target
        self.c = c
        self.minority_class_value = minority_class_value
        self.shuffle = shuffle

    def _validate_estimator(self):
        i = 1
        # nothing to do

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

        self._validate_estimator()

        return self

    def _shuffleTwo(self, a, b):
        #if len(a) != len(b):
        #    raise ValueError("lenth of a ({}) doesn't match length of b ({})".format(len(a), len(b)))

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

        if self.target not in CSS_TARGET:
            raise ValueError('Unknown kind for CSS target.'
                             ' Choices are {}. Got \'{}\' instead.'.format(
                                CSS_TARGET, self.target))

        if self.c < 0 or self.c > 1:
            raise ValueError('Received scaling factor c={}, which'
                             ' is outside the allowed range '
                             '(0-1].'.format(self.c))
        if self.c is 0:
            raise ValueError('Received scaling factor c={}, which is'
                             ' equal to no CSS at.'.format(self.c))

        if self.minority_class_value is not None and \
                not isinstance(self.minority_class_value, int):
            raise ValueError('Unallowed target class value \'{}\'.'
                             ' Valid values include None to automatically'
                             ' infer the target class or any integer number'
                             ' corresponding to the value of the label in y')


        mcv = self.minority_class_value
        if mcv is None:
            # infer minority class value
            counts = Counter(y)
            least_common = counts.most_common()[:-1-1:-1]
            mcv = least_common[0][0]

        # in the following _a is majority, _i is minority
        if self.target is "majority" or self.target is "both":
            col_means_a = np.mean(X[(y != mcv)], axis=0)
            if self.mode is "linear":
                distances_a = abs(np.subtract(X[y != mcv], col_means_a))
        if self.target is "minority" or self.target is "both":
            col_means_i = np.mean(X[(y == mcv)], axis=0)
            if self.mode is "linear":
                distances_i = abs(np.subtract(X[y == mcv], col_means_i))

        if self.target is "majority" or self.target is "both":
            if self.mode is "constant":
                X_scaled_a = X[y != mcv] * (1 - self.c) + col_means_a * self.c
            elif self.mode is "linear":
                scale_factors_mean = (distances_a * self.c)
                scale_factors_values = (1 - self.c * distances_a)

                X_scaled_a = X[y != mcv] * scale_factors_values + col_means_a * scale_factors_mean
        if self.target is "minority" or self.target is "both":
            if self.mode is "constant":
                X_scaled_i = X[y == mcv] * (1 - self.c) + col_means_i * self.c
            elif self.mode is "linear":
                scale_factors_mean = (distances_i * self.c)
                scale_factors_values = (1 - self.c * distances_i)
                X_scaled_i = X[y == mcv] * scale_factors_values + col_means_i * scale_factors_mean

        # merge scaled and non scaled stuff
        if self.target is "majority":
            X_scaled = np.concatenate([X_scaled_a, X[y == mcv]], axis=0)
        elif self.target is "minority":
            X_scaled = np.concatenate([X[y != mcv], X_scaled_i], axis=0)
        else: #"both"
            X_scaled = np.concatenate([X_scaled_a, X_scaled_i], axis=0)

        # make sure that y is in same order like X
        y_assembled = np.concatenate([y[y != mcv], y[y == mcv]], axis=0)

        # shuffle
        X_scaled_shuffled, y_res_shuffled, indices = self._shuffleTwo(X_scaled, y_assembled)

        if self.shuffle:
            return X_scaled_shuffled, y_res_shuffled
        else:
            return X_scaled, y_assembled
