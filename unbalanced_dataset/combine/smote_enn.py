"""Class to perform over-sampling using SMOTE and cleaning using ENN."""
from __future__ import print_function
from __future__ import division

import numpy as np

from numpy import concatenate

from collections import Counter

from ..unbalanced_dataset import UnbalancedDataset

class SMOTEENN(UnbalancedDataset):
    """Class to perform over-sampling using SMOTE and cleaning using ENN.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----

    References
    ----------
    .. [1] G. Batista, R. C. Prati, M. C. Monard. "A study of the behavior of
    several methods for balancing machine learning training data," ACM
    Sigkdd Explorations Newsletter 6 (1), 20-29, 2004.

    """

    def __init__(self, k=5, ratio='auto', random_state=None,
                 size_ngh=3, verbose=True, **kwargs):
        """
        :param size_ngh
            Size of the neighbourhood to consider in order to make
            the comparison between each samples and their NN.

        :param **kwargs
            Parameter to use for the Neareast Neighbours.

        :param k:
            Number of nearest neighbours to use when constructing the synthetic
            samples.

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
        self.size_ngh = size_ngh
        self.kwargs = kwargs

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
        sx, sy = self.make_samples(minx, minx, self.minc, nns,
                                   int(self.ratio * len(miny)),
                                   random_state=self.rs,
                                   verbose=self.verbose)

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx), axis=0)
        ret_y = concatenate((self.y, sy), axis=0)

        # Create a k-NN to fit the whole data
        nn_obj = NearestNeighbors(n_neighbors=self.size_ngh)

        # Fit the whole dataset
        nn_obj.fit(ret_x)

        # Loop over the other classes under picking at random
        for key_idx, key in enumerate(self.ucd.keys()):

            # Get the sample of the current class
            sub_samples_x = ret_x[ret_y == key]
            sub_samples_y = ret_y[ret_y == key]

            # Find the NN for the current class
            nnhood_idx = nn_obj.kneighbors(sub_samples_x,
                                           return_distance=False)

            # Get the label of the corresponding to the index
            nnhood_label = (ret_y[nnhood_idx] == key)

            # Check which one are the same label than the current class
            # Make an AND operation through the k neighbours
            nnhood_bool = np.all(nnhood_label, axis=1)

            # Get the samples which agree all together
            sel_x = np.squeeze(sub_samples_x[np.nonzero(nnhood_bool), :])
            sel_y = sub_samples_y[np.nonzero(nnhood_bool)]

            if key_idx == 0:
                underx = sel_x[:, :]
                undery = sel_y[:]
            else:
                underx = concatenate((underx, sel_x), axis=0)
                undery = concatenate((undery, sel_y), axis=0)

        if self.verbose:
            print("Over-sampling performed: " + str(Counter(undery)))

        return underx, undery
