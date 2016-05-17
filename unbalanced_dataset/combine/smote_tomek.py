"""Class to perform over-sampling using SMOTE and cleaning by removing
Tomek's links."""
from __future__ import print_function
from __future__ import division

import numpy as np

from numpy import concatenate, logical_not

from collections import Counter

from ..unbalanced_dataset import UnbalancedDataset


class SMOTETomek(UnbalancedDataset):
    """Class to perform over-sampling using SMOTE and cleaning by removing
    Tomek's links.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----
    The method is based on [1]_.

    References
    ----------
    .. [1] G. Batista, B. Bazzan, M. Monard, "Balancing Training Data for
       Automated Annotation of Keywords: a Case Study," In WOB, 10-18, 2003.

    """

    def __init__(self, k=5, ratio='auto', random_state=None, verbose=True, **kwargs):
        """
        :param k:
            Number of nearest neighbours to use when constructing the
            synthetic samples.

        :param ratio:
             If 'auto', the ratio will be defined automatically to balanced
            the dataset. If an integer is given, the number of samples
            generated is equal to the number of samples in the minority class
            mulitply by this ratio.

        :param random_state:
            Seed.

        :return:
            The resampled data set with synthetic samples concatenated at the
            end.
        """

        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state,
                                   verbose=verbose)

        # Do not expect any support regarding the selection with this method
        if (kwargs.pop('indices_support', False)):
            raise ValueError('No indices support with this method.')

        # Instance variable to store the number of neighbours to use.
        self.k = k

    def resample(self):

        # Compute the ratio if it is auto
        if self.ratio == 'auto':
            self.ratio = (float(self.ucd[self.maxc] - self.ucd[self.minc]) /
                          float(self.ucd[self.minc]))

        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Finding nns
        # Import the k-NN classifier
        from sklearn.neighbors import NearestNeighbors

        nearest_neighbour = NearestNeighbors(n_neighbors=self.k + 1)
        nearest_neighbour.fit(minx)
        nns = nearest_neighbour.kneighbors(minx, return_distance=False)[:, 1:]

        # Creating synthetic samples
        sx, sy = self.make_samples(minx,
                                   minx,
                                   self.minc,
                                   nns,
                                   int(self.ratio * len(miny)),
                                   random_state=self.rs,
                                   verbose=self.verbose)

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx), axis=0)
        ret_y = concatenate((self.y, sy), axis=0)

        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(ret_x)
        nns = nn.kneighbors(ret_x, return_distance=False)[:, 1]

        # Send the information to is_tomek function to get boolean vector back
        links = self.is_tomek(ret_y, nns, self.minc, self.verbose)

        if self.verbose:
            print("Over-sampling performed:"
                  " " + str(Counter(ret_y[logical_not(links)])))

        # Return data set without majority Tomek links.
        return ret_x[logical_not(links)], ret_y[logical_not(links)]
