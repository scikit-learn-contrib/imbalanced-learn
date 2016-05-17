"""Class performing under-sampling based on the neighbourhood cleaning rule."""
from __future__ import print_function
from __future__ import division

import multiprocessing

import numpy as np

from numpy import concatenate

from random import sample

from collections import Counter

from ..unbalanced_dataset import UnbalancedDataset


class NeighbourhoodCleaningRule(UnbalancedDataset):
    """Class performing under-sampling based on the neighbourhood cleaning
    rule.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----

    References
    ----------
    .. [1] J. Laurikkala, "Improving identification of difficult small classes
       by balancing class distribution," Springer Berlin Heidelberg, 2001.

    """

    def __init__(self, random_state=None, indices_support=False,
                 size_ngh=3, verbose=True, **kwargs):
        """
        :param size_ngh
            Size of the neighbourhood to consider in order to make
            the comparison between each samples and their NN.

        :param **kwargs
            Parameter to use for the Neareast Neighbours.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, random_state=random_state,
                                   indices_support=indices_support,
                                   verbose=verbose)

        # Assign the parameter of the element of this class
        self.size_ngh = size_ngh
        self.n_jobs = kwargs.pop('n_jobs', multiprocessing.cpu_count())
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


        # Import the k-NN classifier
        from sklearn.neighbors import NearestNeighbors

        # Create a k-NN to fit the whole data
        nn_obj = NearestNeighbors(n_neighbors=self.size_ngh,
                                  n_jobs=self.n_jobs)

        # Fit the whole dataset
        nn_obj.fit(self.x)

        idx_to_exclude = []
        # Loop over the other classes under picking at random
        for key in self.ucd.keys():

            # Get the sample of the current class
            sub_samples_x = self.x[self.y == key]

            # Get the samples associated
            idx_sub_sample = np.nonzero(self.y == key)[0]

            # Find the NN for the current class
            nnhood_idx = nn_obj.kneighbors(sub_samples_x, return_distance=False)

            # Get the label of the corresponding to the index
            nnhood_label = (self.y[nnhood_idx] == key)

            # Check which one are the same label than the current class
            # Make an AND operation through the three neighbours
            nnhood_bool = np.logical_not(np.all(nnhood_label, axis=1))

            # If the minority class remove the majority samples (as in politic!!!! ;))
            if key == self.minc:
                # Get the index to exclude
                idx_to_exclude += nnhood_idx[np.nonzero(nnhood_label[np.nonzero(nnhood_bool)])].tolist()
            else:
                # Get the index to exclude
                idx_to_exclude += idx_sub_sample[np.nonzero(nnhood_bool)].tolist()

        idx_to_exclude = np.unique(idx_to_exclude)

        # Create a vector with the sample to select
        sel_idx = np.ones(self.y.shape)
        sel_idx[idx_to_exclude] = 0
        # Exclude as well the minority sample since that they will be
        # concatenated later
        sel_idx[self.y == self.minc] = 0

        # Get the samples from the majority classes
        sel_x = np.squeeze(self.x[np.nonzero(sel_idx), :])
        sel_y = self.y[np.nonzero(sel_idx)]

        # If we need to offer support for the indices selected
        if self.indices_support:
            idx_tmp = np.nonzero(sel_idx)[0]
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
