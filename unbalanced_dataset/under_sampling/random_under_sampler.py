"""Class to perform random under-sampling."""
from __future__ import print_function
from __future__ import division

import numpy as np

from numpy.random import seed
from numpy.random import randint

from random import sample

from collections import Counter

from ..unbalanced_dataset import UnbalancedDataset


class UnderSampler(UnbalancedDataset):
    """Class to perform random under-sampling.

    Object to under sample the majority class(es) by randomly picking samples
    with or without replacement.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----

    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 replacement=True,
                 indices_support=False,
                 verbose=True):
        """
        :param ratio:
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. If an integer is given, the number of samples
            generated is equal to the number of samples in the minority class
            mulitply by this ratio.

        :param random_state:
            Seed.

        :return:
            underx, undery: The features and target values of the under-sampled
            data set.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self,
                                   ratio=ratio,
                                   random_state=random_state,
                                   indices_support=indices_support,
                                   verbose=verbose)

        self.replacement = replacement

    def resample(self):
        """
        ...
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

            # Pick some elements at random
            seed(self.rs)
            if self.replacement:
                indx = randint(low=0, high=self.ucd[key], size=num_samples)
            else:
                indx = sample(range((self.y == key).sum()), num_samples)

            # If we need to offer support for the indices selected
            if self.indices_support:
                idx_tmp = np.nonzero(self.y == key)[0][indx]
                idx_under = np.concatenate((idx_under, idx_tmp), axis=0)

            # Concatenate to the minority class
            underx = concatenate((underx, self.x[self.y == key][indx]), axis=0)
            undery = concatenate((undery, self.y[self.y == key][indx]), axis=0)

        if self.verbose:
            print("Under-sampling performed: " + str(Counter(undery)))

        # Check if the indices of the samples selected should be returned too
        if self.indices_support:
            # Return the indices of interest
            return underx, undery, idx_under
        else:
            return underx, undery
