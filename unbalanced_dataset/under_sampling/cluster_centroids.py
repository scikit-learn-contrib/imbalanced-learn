"""Class to perform under-sampling by generating centroids based on
clustering."""
from __future__ import print_function
from __future__ import division

from numpy import ones
from numpy import concatenate

from collections import Counter

from ..unbalanced_dataset import UnbalancedDataset


class ClusterCentroids(UnbalancedDataset):
    """Class to perform under-sampling by generating centroids base on
    clustering.

    Experimental method that under samples the majority class by replacing a
    cluster of majority samples by the cluster centroid of a KMeans algorithm.
    This algorithm keeps N majority samples by fitting the KMeans algorithm
    with N cluster to the majority class and using the coordinates of the N
    cluster centroids as the new majority samples.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----

    References
    ----------

    """

    def __init__(self, ratio='auto', random_state=None, verbose=True, **kwargs):
        """
        :param kwargs:
            Arguments the user might want to pass to the KMeans object from
            scikit-learn.

        :param ratio:
            The number of cluster to fit with respect to the number of samples
            in the minority class.
            N_clusters = int(ratio * N_minority_samples) = N_maj_undersampled.

        :param random_state:
            Seed.

        :return:
            Under sampled data set.
        """
        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state,
                                   verbose=verbose)

        self.kwargs = kwargs

        # Do not expect any support regarding the selection with this method
        if (self.kwargs.pop('indices_support', False)):
            raise ValueError('No indices support with this method.')

    def resample(self):
        """
        ???

        :return:
        """

        # Compute the ratio if it is auto
        if self.ratio == 'auto':
            self.ratio = 1.

        # Create the clustering object
        from sklearn.cluster import KMeans
        kmeans = KMeans(random_state=self.rs)
        kmeans.set_params(**self.kwargs)

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

        # Loop over the other classes under picking at random
        for key in self.ucd.keys():
            # If the minority class is up, skip it.
            if key == self.minc:
                continue

            # Set the number of clusters to be no more than the number of
            # samples
            if self.ratio * self.ucd[self.minc] > self.ucd[key]:
                n_clusters = self.ucd[key]
            else:
                n_clusters = int(self.ratio * self.ucd[self.minc])

            # Set the number of clusters and find the centroids
            kmeans.set_params(n_clusters=n_clusters)
            kmeans.fit(self.x[self.y == key])
            centroids = kmeans.cluster_centers_

            # Concatenate to the minority class
            underx = concatenate((underx, centroids), axis=0)
            undery = concatenate((undery, ones(n_clusters) * key), axis=0)

        if self.verbose:
            print("Under-sampling performed: " + str(Counter(undery)))

        return underx, undery
