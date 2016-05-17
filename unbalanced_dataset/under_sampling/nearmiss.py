"""Class to perform under-sampling based on nearmiss methods."""
from __future__ import print_function
from __future__ import division

import multiprocessing

import numpy as np

from numpy import concatenate

from collections import Counter

from ..unbalanced_dataset import UnbalancedDataset


class NearMiss(UnbalancedDataset):
    """Class to perform under-sampling based on NearMiss methods.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----
    The methods are based on [1]_.

    References
    ----------
    .. [1] I. Mani, I. Zhang. "kNN approach to unbalanced data distributions:
       a case study involving information extraction," In Proceedings of
       workshop on learning from imbalanced datasets, 2003.

    """

    def __init__(self, ratio='auto', random_state=None,
                 version=1, size_ngh=3, ver3_samp_ngh=3,
                 indices_support=False, verbose=True, **kwargs):
        """
        :param version:
            Version of the NearMiss to use. Possible values
            are 1, 2 or 3. See the original paper for details
            about these different versions.

        :param size_ngh:
            Size of the neighbourhood to consider to compute the
            average distance to the minority point samples.

        :param ver3_samp_ngh:
            NearMiss-3 algorithm start by a phase of re-sampling. This
            parameter correspond to the number of neighbours selected
            create the sub_set in which the selection will be performed.

        :param **kwargs:
            Parameter to use for the Nearest Neighbours.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state,
                                   indices_support=indices_support,
                                   verbose=verbose)

        # Assign the parameter of the element of this class
        # Check that the version asked is implemented
        if not (version == 1 or version == 2 or version == 3):
            raise ValueError('UnbalancedData.NearMiss: there is only 3 '
                             'versions available with parameter version=1/2/3')

        self.version = version
        self.size_ngh = size_ngh
        self.ver3_samp_ngh = ver3_samp_ngh
        self.n_jobs = kwargs.pop('n_jobs', multiprocessing.cpu_count())
        self.kwargs = kwargs

    def resample(self):
        """
        """

        # Compute the ratio if it is auto
        if self.ratio == 'auto':
            self.ratio = 1.

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

        # If we need to offer support for the indices
        if self.indices_support:
            idx_under = np.nonzero(self.y == self.minc)[0]

        # For each element of the current class, find the set of NN
        # of the minority class
        from sklearn.neighbors import NearestNeighbors

        # Call the constructor of the NN
        nn_obj = NearestNeighbors(n_neighbors=self.size_ngh,
                                  n_jobs=self.n_jobs,
                                  **self.kwargs)

        # Fit the minority class since that we want to know the distance
        # to these point
        nn_obj.fit(self.x[self.y == self.minc])

        # Loop over the other classes under picking at random
        for key in self.ucd.keys():

            # If the minority class is up, skip it
            if key == self.minc:
                continue

            # Set the ratio to be no more than the number of samples available
            if self.ratio * self.ucd[self.minc] > self.ucd[key]:
                num_samples = self.ucd[key]
            else:
                num_samples = int(self.ratio * self.ucd[self.minc])

            # Get the samples corresponding to the current class
            sub_samples_x = self.x[self.y == key]
            sub_samples_y = self.y[self.y == key]

            if self.version == 1:
                # Find the NN
                dist_vec, idx_vec = nn_obj.kneighbors(sub_samples_x,
                                                      n_neighbors=self.size_ngh)

                # Select the right samples
                sel_x, sel_y, idx_tmp = self.__SelectionDistBased__(dist_vec,
                                                                    num_samples,
                                                                    key,
                                                                    sel_strategy='nearest')
            elif self.version == 2:
                # Find the NN
                dist_vec, idx_vec = nn_obj.kneighbors(sub_samples_x,
                                                      n_neighbors=self.y[self.y == self.minc].size)

                # Select the right samples
                sel_x, sel_y, idx_tmp = self.__SelectionDistBased__(dist_vec,
                                                                    num_samples,
                                                                    key,
                                                                    sel_strategy='nearest')
            elif self.version == 3:
                # We need a new NN object to fit the current class
                nn_obj_cc = NearestNeighbors(n_neighbors=self.ver3_samp_ngh,
                                             n_jobs=self.n_jobs,
                                             **self.kwargs)
                nn_obj_cc.fit(sub_samples_x)

                # Find the set of NN to the minority class
                dist_vec, idx_vec = nn_obj_cc.kneighbors(self.x[self.y == self.minc])

                # Create the subset containing the samples found during the NN
                # search. Linearize the indexes and remove the double values
                idx_vec = np.unique(idx_vec.reshape(-1))

                # Create the subset
                sub_samples_x = sub_samples_x[idx_vec, :]
                sub_samples_y = sub_samples_y[idx_vec]

                # Compute the NN considering the current class
                dist_vec, idx_vec = nn_obj.kneighbors(sub_samples_x,
                                                      n_neighbors=self.size_ngh)

                sel_x, sel_y, idx_tmp = self.__SelectionDistBased__(dist_vec,
                                                                    num_samples,
                                                                    key,
                                                                    sel_strategy='farthest')

            # If we need to offer support for the indices selected
            if self.indices_support:
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


    def __SelectionDistBased__(self,
                               dist_vec,
                               num_samples,
                               key,
                               sel_strategy='nearest'):

        # Compute the distance considering the farthest neighbour
        dist_avg_vec = np.sum(dist_vec[:, -self.size_ngh:], axis=1)

        # Sort the list of distance and get the index
        if sel_strategy == 'nearest':
            sort_way = False
        elif sel_strategy == 'farthest':
            sort_way = True
        else:
            raise ValueError('Unbalanced.NearMiss: the sorting can be done '
                             'only with nearest or farthest data points.')

        sorted_idx = sorted(range(len(dist_avg_vec)),
                            key=dist_avg_vec.__getitem__,
                            reverse=sort_way)

        # Select the desired number of samples
        sel_idx = sorted_idx[:num_samples]

        return self.x[self.y == key][sel_idx], self.y[self.y == key][sel_idx], np.nonzero(self.y == key)[0][sel_idx]
