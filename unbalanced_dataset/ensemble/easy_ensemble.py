"""Class to perform under-sampling using easy ensemble."""
from __future__ import print_function

import numpy as np

from ..under_sampling.under_sampler import UnderSampler


class EasyEnsemble(UnderSampler):
    """Perform under-sampling using an ensemble of random selection.

    This method iteratively select subset and make an ensemble of the
    different sets.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----
    The method is described in [1]_.

    References
    ----------
    .. [1] X. Y. Liu, J. Wu and Z. H. Zhou, "Exploratory Undersampling for
       Class-Imbalance Learning," in IEEE Transactions on Systems, Man, and
       Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539-550,
       April 2009.

    """

    def __init__(self, ratio='auto', random_state=None, replacement=False,
                 n_subsets=10, verbose=True):
        """
        :param ratio:
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. If an integer is given, the number of samples
            generated is equal to the number of samples in the minority class
            mulitply by this raio.

        :param random_state:
            Seed.

        :param replacement:
            Either or not to sample randomly with replacement or not.

        :param n_subsets:
            Number of subsets to generate.
        """

        # Passes the relevant parameters back to the parent class.
        UnderSampler.__init__(self, ratio=ratio,
                              random_state=random_state,
                              replacement=replacement,
                              verbose=verbose)

        self.n_subsets = n_subsets

    def resample(self):
        """
        :return subsets_x:
            Python list containing the different data arrays generated and
            balanced.

        :return subsets_y:
            Python list containing the different label arrays generated and
            balanced.
        """

        subsets_x = []
        subsets_y = []

        for s in range(self.n_subsets):
            if self.verbose:
                print("Creation of the set #%i" % s)

            tmp_subset_x, tmp_subset_y = UnderSampler.resample(self)
            subsets_x.append(tmp_subset_x)
            subsets_y.append(tmp_subset_y)

        return np.array(subsets_x), np.array(subsets_y)
