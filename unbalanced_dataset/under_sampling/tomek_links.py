"""Class to perform under-sampling by removing Tomek's links."""
from __future__ import print_function
from __future__ import division

import multiprocessing

import numpy as np

from numpy import logical_not

from collections import Counter

from ..unbalanced_dataset import UnbalancedDataset


class TomekLinks(UnbalancedDataset):
    """Class to perform under-sampling by removing Tomek's links.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----

    References
    ----------

    """

    def __init__(self, indices_support=False, verbose=True, **kwargs):
        """
        No parameters.

        :return:
            Nothing.
        """

        UnbalancedDataset.__init__(self, indices_support, verbose=verbose)

        self.n_jobs = kwargs.pop('n_jobs', multiprocessing.cpu_count())
        
    def resample(self):
        """
        :return:
            Return the data with majority samples that form a Tomek link
            removed.
        """

        from sklearn.neighbors import NearestNeighbors

        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        nn.fit(self.x)
        nns = nn.kneighbors(self.x, return_distance=False)[:, 1]

        # Send the information to is_tomek function to get boolean vector back
        if self.verbose:
            print("Looking for majority Tomek links...")
        links = self.is_tomek(self.y, nns, self.minc, self.verbose)

        if self.verbose:
            print("Under-sampling "
                  "performed: " + str(Counter(self.y[logical_not(links)])))

        # Check if the indices of the samples selected should be returned too
        if self.indices_support:
            # Return the indices of interest
            return self.x[logical_not(links)], self.y[logical_not(links)], np.nonzero(logical_not(links))[0]
        else:
            # Return data set without majority Tomek links.
            return self.x[logical_not(links)], self.y[logical_not(links)]

    @staticmethod
    def is_tomek(y, nn_index, class_type, verbose=True):
        """
        is_tomek uses the target vector and the first neighbour of every sample
        point and looks for Tomek pairs. Returning a boolean vector with True
        for majority Tomek links.

        :param y:
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority or not

        :param nn_index:
            The index of the closes nearest neighbour to a sample point.

        :param class_type:
            The label of the minority class.

        :return:
            Boolean vector on len( # samples ), with True for majority samples
            that are Tomek links.
        """

        # Initialize the boolean result as false, and also a counter
        links = np.zeros(len(y), dtype=bool)
        count = 0

        # Loop through each sample and looks whether it belongs to the minority
        # class. If it does, we don't consider it since we want to keep all
        # minority samples. If, however, it belongs to the majority sample we
        # look at its first neighbour. If its closest neighbour also has the
        # current sample as its closest neighbour, the two form a Tomek link.
        for ind, ele in enumerate(y):

            if ele == class_type:
                continue

            if y[nn_index[ind]] == class_type:

                # If they form a tomek link, put a True marker on this
                # sample, and increase counter by one.
                if nn_index[nn_index[ind]] == ind:
                    links[ind] = True
                    count += 1

        if verbose:
            print("%i Tomek links found." % count)

        return links
