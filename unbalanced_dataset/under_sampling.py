from __future__ import print_function
from __future__ import division
import numpy as np
from numpy import logical_not, ones
from numpy.random import seed, randint
from numpy import concatenate
from random import sample
from collections import Counter
from .unbalanced_dataset import UnbalancedDataset


class UnderSampler(UnbalancedDataset):
    """
    Object to under sample the majority class(es) by randomly picking samples
    with or without replacement.
    """

    def __init__(self,
                 ratio=1.,
                 random_state=None,
                 replacement=True,
                 verbose=True):
        """
        :param ratio:
            The ratio of majority elements to sample with respect to the number
            of minority cases.

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
                                   verbose=verbose)

        self.replacement = replacement

    def resample(self):
        """
        ...
        """

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

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

            # Concatenate to the minority class
            underx = concatenate((underx, self.x[self.y == key][indx]), axis=0)
            undery = concatenate((undery, self.y[self.y == key][indx]), axis=0)

        if self.verbose:
            print("Under-sampling performed: " + str(Counter(undery)))

        return underx, undery


class TomekLinks(UnbalancedDataset):
    """
    Object to identify and remove majority samples that form a Tomek link with
    minority samples.
    """

    def __init__(self, verbose=True):
        """
        No parameters.

        :return:
            Nothing.
        """

        UnbalancedDataset.__init__(self, verbose=verbose)

    def resample(self):
        """
        :return:
            Return the data with majority samples that form a Tomek link
            removed.
        """

        from sklearn.neighbors import NearestNeighbors

        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(self.x)
        nns = nn.kneighbors(self.x, return_distance=False)[:, 1]

        # Send the information to is_tomek function to get boolean vector back
        if self.verbose:
            print("Looking for majority Tomek links...")
        links = self.is_tomek(self.y, nns, self.minc, self.verbose)

        if self.verbose:
            print("Under-sampling "
                  "performed: " + str(Counter(self.y[logical_not(links)])))

        # Return data set without majority Tomek links.
        return self.x[logical_not(links)], self.y[logical_not(links)]


class ClusterCentroids(UnbalancedDataset):
    """
    Experimental method that under samples the majority class by replacing a
    cluster of majority samples by the cluster centroid of a KMeans algorithm.

    This algorithm keeps N majority samples by fitting the KMeans algorithm
    with N cluster to the majority class and using the coordinates of the N
    cluster centroids as the new majority samples.
    """

    def __init__(self, ratio=1, random_state=None, verbose=True, **kwargs):
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

    def resample(self):
        """
        ???

        :return:
        """

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


class NearMiss(UnbalancedDataset):
    """
    An implementation of NearMiss.

    See the original paper: NearMiss - "kNN Approach to Unbalanced Data
    Distributions: A Case Study involving Information Extraction" by Zhang
    et al. for more details.
    """

    def __init__(self, ratio=1., random_state=None,
                 version=1, size_ngh=3, ver3_samp_ngh=3,
                 verbose=True, **kwargs):
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
                                   verbose=verbose)

        # Assign the parameter of the element of this class
        # Check that the version asked is implemented
        if not (version == 1 or version == 2 or version == 3):
            raise ValueError('UnbalancedData.NearMiss: there is only 3 '
                             'versions available with parameter version=1/2/3')

        self.version = version
        self.size_ngh = size_ngh
        self.ver3_samp_ngh = ver3_samp_ngh
        self.kwargs = kwargs

    def resample(self):
        """
        """

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

        # For each element of the current class, find the set of NN
        # of the minority class
        from sklearn.neighbors import NearestNeighbors

        # Call the constructor of the NN
        nn_obj = NearestNeighbors(n_neighbors=self.size_ngh, **self.kwargs)

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
                sel_x, sel_y = self.__SelectionDistBased__(dist_vec,
                                                           num_samples,
                                                           key,
                                                           sel_strategy='nearest')
            elif self.version == 2:
                # Find the NN
                dist_vec, idx_vec = nn_obj.kneighbors(sub_samples_x,
                                                      n_neighbors=self.y[self.y == self.minc].size)

                # Select the right samples
                sel_x, sel_y = self.__SelectionDistBased__(dist_vec,
                                                           num_samples,
                                                           key,
                                                           sel_strategy='nearest')
            elif self.version == 3:
                # We need a new NN object to fit the current class
                nn_obj_cc = NearestNeighbors(n_neighbors=self.ver3_samp_ngh,
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

                sel_x, sel_y = self.__SelectionDistBased__(dist_vec,
                                                           num_samples,
                                                           key,
                                                           sel_strategy='farthest')

            underx = concatenate((underx, sel_x), axis=0)
            undery = concatenate((undery, sel_y), axis=0)

        if self.verbose:
            print("Under-sampling performed: " + str(Counter(undery)))

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

        return self.x[self.y == key][sel_idx], self.y[self.y == key][sel_idx]


class CondensedNearestNeighbour(UnbalancedDataset):
    """
    An implementation of Condensend Neareat Neighbour.

    See the original paper: CNN - "Addressing the Curse of Imbalanced Training
    Set: One-Sided Selection" by Khubat et al. for more details.
    """

    def __init__(self, random_state=None,
                 size_ngh=1, n_seeds_S=1, verbose=True,
                 **kwargs):
        """

        :param size_ngh
            Size of the neighbourhood to consider to compute the
            average distance to the minority point samples.

        :param n_seeds_S
            Number of samples to extract in order to build the set S.

        :param **kwargs
            Parameter to use for the Neareast Neighbours.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, random_state=random_state,
                                   verbose=verbose)

        # Assign the parameter of the element of this class
        self.size_ngh = size_ngh
        self.n_seeds_S = n_seeds_S
        self.kwargs = kwargs

    def resample(self):
        """
        """

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

        # Import the K-NN classifier
        from sklearn.neighbors import KNeighborsClassifier

        # Loop over the other classes under picking at random
        for key in self.ucd.keys():

            # If the minority class is up, skip it
            if key == self.minc:
                continue

            # Randomly get one sample from the majority class
            maj_sample = sample(self.x[self.y == key],
                                self.n_seeds_S)

            # Create the set C
            C_x = np.append(self.x[self.y == self.minc],
                            maj_sample,
                            axis=0)
            C_y = np.append(self.y[self.y == self.minc],
                            [key] * self.n_seeds_S)

            # Create the set S
            S_x = self.x[self.y == key]
            S_y = self.y[self.y == key]

            # Create a k-NN classifier
            knn = KNeighborsClassifier(n_neighbors=self.size_ngh,
                                       **self.kwargs)

            # Fit C into the knn
            knn.fit(C_x, C_y)

            # Classify on S
            pred_S_y = knn.predict(S_x)

            # Find the misclassified S_y
            sel_x = np.squeeze(S_x[np.nonzero(pred_S_y != S_y), :])
            sel_y = S_y[np.nonzero(pred_S_y != S_y)]

            underx = concatenate((underx, sel_x), axis=0)
            undery = concatenate((undery, sel_y), axis=0)

        if self.verbose:
            print("Under-sampling performed: " + str(Counter(undery)))

        return underx, undery


class OneSidedSelection(UnbalancedDataset):
    """
    An implementation of One-Sided Selection.

    See the original paper: OSS - "Addressing the Curse of Imbalanced Training
    Set: One-Sided Selection" by Khubat et al. for more details.
    """

    def __init__(self, random_state=None,
                 size_ngh=1, n_seeds_S=1, verbose=True,
                 **kwargs):
        """

        :param size_ngh
            Size of the neighbourhood to consider to compute the
            average distance to the minority point samples.

        :param n_seeds_S
            Number of samples to extract in order to build the set S.

        :param **kwargs
            Parameter to use for the Neareast Neighbours.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, random_state=random_state,
                                   verbose=verbose)

        # Assign the parameter of the element of this class
        self.size_ngh = size_ngh
        self.n_seeds_S = n_seeds_S
        self.kwargs = kwargs

    def resample(self):
        """
        """

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

        # Import the K-NN classifier
        from sklearn.neighbors import KNeighborsClassifier

        # Loop over the other classes under picking at random
        for key in self.ucd.keys():

            # If the minority class is up, skip it
            if key == self.minc:
                continue

            # Randomly get one sample from the majority class
            maj_sample = sample(self.x[self.y == key],
                                self.n_seeds_S)

            # Create the set C
            C_x = np.append(self.x[self.y == self.minc],
                            maj_sample,
                            axis=0)
            C_y = np.append(self.y[self.y == self.minc],
                            [key] * self.n_seeds_S)

            # Create the set S
            S_x = self.x[self.y == key]
            S_y = self.y[self.y == key]

            # Create a k-NN classifier
            knn = KNeighborsClassifier(n_neighbors=self.size_ngh,
                                       **self.kwargs)

            # Fit C into the knn
            knn.fit(C_x, C_y)

            # Classify on S
            pred_S_y = knn.predict(S_x)

            # Find the misclassified S_y
            sel_x = np.squeeze(S_x[np.nonzero(pred_S_y != S_y), :])
            sel_y = S_y[np.nonzero(pred_S_y != S_y)]

            underx = concatenate((underx, sel_x), axis=0)
            undery = concatenate((undery, sel_y), axis=0)

        from sklearn.neighbors import NearestNeighbors

        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(underx)
        nns = nn.kneighbors(underx, return_distance=False)[:, 1]

        # Send the information to is_tomek function to get boolean vector back
        if self.verbose:
            print("Looking for majority Tomek links...")
        links = self.is_tomek(undery, nns, self.minc, self.verbose)

        if self.verbose:
            print("Under-sampling "
                  "performed: " + str(Counter(undery[logical_not(links)])))

        # Return data set without majority Tomek links.
        return underx[logical_not(links)], undery[logical_not(links)]


class NeighbourhoodCleaningRule(UnbalancedDataset):
    """
    An implementation of Neighboorhood Cleaning Rule.

    See the original paper: NCL - "Improving identification of difficult small
    classes by balancing class distribution" by Laurikkala et al. for more details.
    """

    def __init__(self, random_state=None,
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
                                   verbose=verbose)

        # Assign the parameter of the element of this class
        self.size_ngh = size_ngh
        self.kwargs = kwargs

    def resample(self):
        """
        """

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

        # Import the k-NN classifier
        from sklearn.neighbors import NearestNeighbors

        # Create a k-NN to fit the whole data
        nn_obj = NearestNeighbors(n_neighbors=self.size_ngh)

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

        # Create a vector with the sample to select
        sel_idx = np.ones(self.y.shape)
        sel_idx[idx_to_exclude] = 0

        # Get the samples from the majority classes
        sel_x = np.squeeze(self.x[np.nonzero(sel_idx), :])
        sel_y = self.y[np.nonzero(sel_idx)]

        underx = concatenate((underx, sel_x), axis=0)
        undery = concatenate((undery, sel_y), axis=0)

        if self.verbose:
            print("Under-sampling performed: " + str(Counter(undery)))

        return underx, undery
