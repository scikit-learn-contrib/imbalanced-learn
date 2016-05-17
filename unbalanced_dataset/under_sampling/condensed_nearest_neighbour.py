"""Class to perform under-sampling based on the condensed nearest neighbour
method."""
from __future__ import print_function
from __future__ import division

import multiprocessing

import numpy as np

from numpy import logical_not, ones
from numpy.random import seed, randint
from numpy import concatenate

from random import sample

from collections import Counter

from ..unbalanced_dataset import UnbalancedDataset


class CondensedNearestNeighbour(UnbalancedDataset):
    """Class to perform under-sampling based on the condensed nearest neighbour
    method.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----
    The method is based on [1]_.

    References
    ----------
    .. [1] M. Kubat, S. Matwin, "Addressing the curse of imbalanced training
       sets: one-sided selection," In ICML, vol. 97, pp. 179-186, 1997.

    """

    def __init__(self, random_state=None, indices_support=False,
                 size_ngh=1, n_seeds_S=1, verbose=True,
                 **kwargs):
        """

        :param size_ngh
            Size of the neighbourhood to consider to compute the
            average distance to the minority point samples.

        :param n_seeds_S
            Number of samples to extract in order to build the set S.

        :param **kwargs
            Parameter to use for the Neareast Neighbours.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, random_state=random_state,
                                   indices_support=indices_support, verbose=verbose)

        # Assign the parameter of the element of this class
        self.size_ngh = size_ngh
        self.n_seeds_S = n_seeds_S
        self.kwargs = kwargs

    def resample(self):
        """
        """

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

        # If we need to offer support for the indices
        if self.indices_support:
            idx_under = np.nonzero(self.y == self.minc)[0]

        # Import the K-NN classifier
        from sklearn.neighbors import KNeighborsClassifier

        # Loop over the other classes under picking at random
        for key in self.ucd.keys():

            # If the minority class is up, skip it
            if key == self.minc:
                continue

            # Randomly get one sample from the majority class
            maj_sample = sample(self.x[self.y == key],
                                self.n_seeds_S)

            # Create the set C
            C_x = np.append(self.x[self.y == self.minc],
                            maj_sample,
                            axis=0)
            C_y = np.append(self.y[self.y == self.minc],
                            [key] * self.n_seeds_S)

            # Create the set S
            S_x = self.x[self.y == key]
            S_y = self.y[self.y == key]

            # Create a k-NN classifier
            knn = KNeighborsClassifier(n_neighbors=self.size_ngh,
                                       **self.kwargs)

            # Fit C into the knn
            knn.fit(C_x, C_y)

            # Classify on S
            pred_S_y = knn.predict(S_x)

            # Find the misclassified S_y
            sel_x = np.squeeze(S_x[np.nonzero(pred_S_y != S_y), :])
            sel_y = S_y[np.nonzero(pred_S_y != S_y)]

            # If we need to offer support for the indices selected
            if self.indices_support:
                idx_tmp = np.nonzero(self.y == key)[0][np.nonzero(pred_S_y != S_y)]
                idx_under = np.concatenate((idx_under, idx_tmp), axis=0)

            underx = concatenate((underx, sel_x), axis=0)
            undery = concatenate((undery, sel_y), axis=0)

        if self.verbose:
            print("Under-sampling performed: " + str(Counter(undery)))

        # Check if the indices of the samples selected should be returned too
        if self.indices_support:
            # Return the indices of interest
            return underx, undery, idx_under
        else:
            return underx, undery
