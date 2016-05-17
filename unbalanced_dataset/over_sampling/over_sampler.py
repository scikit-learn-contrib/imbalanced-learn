"""Class to perform random over-sampling."""
from __future__ import print_function
from __future__ import division

import numpy as np
from numpy.random import seed
from numpy.random import randint
from numpy import concatenate
from numpy import asarray

from collections import Counter

from ..unbalanced_dataset import UnbalancedDataset


class OverSampler(UnbalancedDataset):
    """Class to perform random over-sampling.

    Object to over-sample the minority class(es) by picking samples at random
    with replacement.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----
    Supports multiple classes.
    """

    def __init__(self, ratio='auto', method='replacement', random_state=None, verbose=True, **kwargs):
        """
        :param ratio:
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. If an integer is given, the number of samples
            generated is equal to the number of samples in the minority class
            mulitply by this ratio.

        :param random_state:
            Seed.

        :return:
            Nothing.
        """
        UnbalancedDataset.__init__(self,
                                   ratio=ratio,
                                   random_state=random_state,
                                   verbose=verbose)

        # Do not expect any support regarding the selection with this method
        if (kwargs.pop('indices_support', False)):
            raise ValueError('No indices support with this method.')

        self.method = method
        if (self.method == 'gaussian-perturbation'):
            self.mean_gaussian = kwargs.pop('mean_gaussian', 0.0)
            self.std_gaussian = kwargs.pop('std_gaussian', 1.0)

    def resample(self):
        """
        Over samples the minority class by randomly picking samples with
        replacement.

        :return:
            overx, overy: The features and target values of the over-sampled
            data set.
        """

        # Compute the ratio if it is auto
        if self.ratio == 'auto':
            self.ratio = (float(self.ucd[self.maxc] - self.ucd[self.minc]) /
                          float(self.ucd[self.minc]))

        # Start with the majority class
        overx = self.x[self.y == self.maxc]
        overy = self.y[self.y == self.maxc]

        # Loop over the other classes over picking at random
        for key in self.ucd.keys():
            if key == self.maxc:
                continue

            # If the ratio given is too large such that the minority becomes a
            # majority, clip it.
            if self.ratio * self.ucd[key] > self.ucd[self.maxc]:
                num_samples = self.ucd[self.maxc] - self.ucd[key]
            else:
                num_samples = int(self.ratio * self.ucd[key])

            if (self.method == 'replacement'):
                # Pick some elements at random
                seed(self.rs)
                indx = randint(low=0, high=self.ucd[key], size=num_samples)

                # Concatenate to the majority class
                overx = concatenate((overx,
                                     self.x[self.y == key],
                                     self.x[self.y == key][indx]), axis=0)

                overy = concatenate((overy,
                                     self.y[self.y == key],
                                     self.y[self.y == key][indx]), axis=0)

            elif (self.method == 'gaussian-perturbation'):
                # Pick the index of the samples which will be modified
                seed(self.rs)
                indx = randint(low=0, high=self.ucd[key], size=num_samples)

                # Generate the new samples
                sam_pert = []
                for i in indx:
                    pert = np.random.normal(self.mean_gaussian, self.std_gaussian, self.x[self.y == key][i])
                    sam_pert.append(self.x[self.y == key][i] + pert)

                # Convert the list to numpy array
                sam_pert = np.array(sam_pert)

                # Concatenate to the majority class
                overx = concatenate((overx,
                                     self.x[self.y == key],
                                     sam_pert), axis=0)

                overy = concatenate((overy,
                                     self.y[self.y == key],
                                     self.y[self.y == key][indx]), axis=0)

        if self.verbose:
            print("Over-sampling performed: " + str(Counter(overy)))

        # Return over sampled dataset
        return overx, overy
