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
