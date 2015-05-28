"""
UnbalancedDataset
=================

UnbalancedDataset is a python module offering a number of re-sampling
techniques commonly used in datasets showing strong between-class
imbalance.

Most classification algorithms will only perform optimally when the number of
samples of each class is roughly the same. Highly skewed datasets, where the
minority heavily outnumbered by one or more classes, haven proven to be a
challenge while at the same time becoming more and more common.

One way of addresing this issue is by re-sampling the dataset as to offset this
imbalance with the hope of arriving and a more robust and fair decision
boundary than you would otherwise.

Resampling techniques are divided in two categories:
    1. Under-sampling the majority class(es).
    2. Over-sampling the minority class.

Bellow is a list of the methods currently implemented in this module.

* Under-sampling
    1. Random majority under-sampling with replacement
    2. Extraction of majority-minority Tomek links
    3. Under-sampling with Cluster Centroids
    4. NearMiss-(1 & 2 & 3)
    5. Condensend Nearest Neighbour
    6. One-Sided Selection
    7. Neighboorhood Cleaning Rule

* Over-sampling
    1. Random minority over-sampling with replacement
    2. SMOTE - Synthetic Minority Over-sampling Technique
    3. bSMOTE(1&2) - Borderline SMOTE of types 1 and 2
    4. SVM_SMOTE - Support Vectors SMOTE

* Over-sampling follow by under-sampling
    1. SMOTE + Tomek links
    2. SMOTE + ENN

* Ensemble sampling
    1. EasyEnsemble
    2. BalanceCascade

This is a work in progress. Any comments, suggestions or corrections are
welcome.

References:

[1] SMOTE - "SMOTE: synthetic minority over-sampling technique" by Chawla,
N.V et al.

[2] Borderline SMOTE -  "Borderline-SMOTE: A New Over-Sampling Method in
Imbalanced Data Sets Learning, Hui Han, Wen-Yuan Wang, Bing-Huan Mao"

[3] SVM_SMOTE - "Borderline Over-sampling for Imbalanced Data Classification,
Nguyen, Cooper, Kamei"

[4] NearMiss - "kNN approach to unbalanced data distributions: A case study involving information extraction" by Zhang et al.

[5] CNN - "Addressing the Curse of Imbalanced Training Sets: One-Sided Selection" by Kubat et al.

[6] One-Sided Selection - "Addressing the Curse of Imbalanced Training Sets: One-Sided Selection" by Kubat et al.

[7] NCL - "Improving identification of difficult small classes by balancing class distribution" by Laurikkala et al.

[8] SMOTE + Tomek - "Balancing training data for automated annotation of keywords: a case study" by Batista et al.

[9] SMOTE + ENN - "A study of the behavior of several methods for balancing machine learning training data" by Batista et al.

[10] EasyEnsemble & BalanceCascade - "Exploratory Understanding for Class-Imbalance Learning" by Liu et al.

TO DO LIST:
===========
    1. Turn global functions into static methods
    2. Add more comments
"""

from __future__ import division
from __future__ import print_function

__author__ = 'fnogueira, glemaitre'

from random import gauss, sample, betavariate
from random import seed as pyseed
import numpy as np
from numpy.random import seed, randint, uniform
from numpy import zeros, ones, concatenate, logical_not, asarray
from numpy import sum as nsum
from collections import Counter


# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
#                                Parent Class!
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
class UnbalancedDataset(object):
    """
    Parent class with the main methods: fit, transform and fit_transform
    """

    def __init__(self, ratio=1., random_state=None, verbose=True):
        """
        Initialize this object and its instance variables.

        :param ratio:
            ratio will be used in different ways for different children object.
            But in general it quantifies the amount of under sampling or over
            sampling to be perfomed with respect to the number of samples
            present in the minority class.

        :param random_state:
            Seed for random number generation.

        :param verbose:
            Boolean to either or not print information about the processing

        :return:
            Nothing.


        Instance variables:
        -------------------

        :self.ratio:
            Holds the ratio parameter.

        :self.rs:
            Holds the seed for random state

        :self.x:
            Holds the feature matrix.

        :self.y:
            Holds the target vector.

        :self.minc:
            Store the label of the minority class.

        :self.maxc:
            Store the label of the majority class.

        :self.ucd:
            Dictionary to hold the label of all the class and the number of
            elements in each.
            {'label A' : #a, 'label B' : #b, ...}

        :self.verbose:
            Boolean allowing some verbosing during the processing.
        """

        ##
        self.ratio = ratio
        self.rs = random_state

        ##
        self.x = None
        self.y = None

        ##
        self.minc = None
        self.maxc = None
        self.ucd = {}

        ##
        self.out_x = None
        self.out_y = None

        #
        self.num = None

        #
        self.verbose = verbose

    def resample(self):
        pass

    def fit(self, x, y):
        """
        Class method to find the relevant class statistics and store it.

        :param x:
            Features.

        :param y:
            Target values.

        :return:
            Nothing
        """

        self.x = x
        self.y = y

        if self.verbose==True:
            print("Determining classes statistics... ", end="")

        # Get all the unique elements in the target array
        uniques = set(self.y)

        # something#
        if len(uniques) == 1:
            raise RuntimeError("Only one class detected, aborting...")

        self.num = zeros((len(uniques), 2))

        # Create a dictionary to store the statistic for each element
        for elem in uniques:
            self.ucd[elem] = 0

        # Populate this dictionary with the class proportions
        for elem in self.y:
            self.ucd[elem] += 1

        ## ----- ##
        # Find the minority and majority classes
        curre_min = len(y)
        curre_max = 0

        # something ...#
        for key in self.ucd.keys():

            if self.ucd[key] < curre_min:
                self.minc = key
                curre_min = self.ucd[key]

            if self.ucd[key] > curre_max:
                self.maxc = key
                curre_max = self.ucd[key]

        if self.verbose==True:
            print(str(len(uniques)) + 
                  " classes detected: " + 
                  str(self.ucd), end="\n")

    def transform(self):
        """
        Class method to re-sample the dataset with a particular technique.

        :return:
            The re-sampled data set.
        """
        
        if self.verbose==True:
            print("Start resampling ...")
        
        self.out_x, self.out_y = self.resample()

        return self.out_x, self.out_y

    def fit_transform(self, x, y):
        """
        Class method to fit and transform the data set automatically.

        :param x:
            Features.

        :param y:
            Target values.

        :return:
            The re-sampled data set.
        """

        self.fit(x, y)
        self.out_x, self.out_y = self.resample()

        return self.out_x, self.out_y

    # ----------------------------------- // ----------------------------------- #
    #                                Static Methods
    # ----------------------------------- // ----------------------------------- #
    @staticmethod
    def is_tomek(y, nn_index, class_type, verbose=True):
        """
        is_tomek uses the target vector and the first neighbour of every sample
        point and looks for Tomek pairs. Returning a boolean vector with True for
        majority Tomek links.

        :param y:
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority or not

        :param nn_index:
            The index of the closes nearest neighbour to a sample point.

        :param class_type:
            The label of the minority class.

        :return:
            Boolean vector on len( # samples ), with True for majority samples that
            are Tomek links.
        """

        # Initialize the boolean result as false, and also a counter
        links = zeros(len(y), dtype=bool)
        count = 0

        # Loop through each sample and looks whether it belongs to the minority
        # class. If it does, we don't consider it since we want to keep all
        # minority samples. If, however, it belongs to the majority sample we look
        # at its first neighbour. If its closest neighbour also has the current
        # sample as its closest neighbour, the two form a Tomek link.
        for ind, ele in enumerate(y):

            if ele == class_type:
                continue

            if y[nn_index[ind]] == class_type:

                # If they form a tomek link, put a True marker on this sample, and
                # increase counter by one.
                if nn_index[nn_index[ind]] == ind:
                    links[ind] = True
                    count += 1
                    
        if verbose==True:
            print("%i Tomek links found." % count)

        return links

    @staticmethod
    def make_samples(x, nn_data, y_type, nn_num, n_samples, 
                     step_size=1., random_state=None, verbose=True):
        """
        A support function that returns artificial samples constructed along the
        line connecting nearest neighbours.

        :param x:
            Minority points for which new samples are going to be created.

        :param nn_data:
            Data set carrying all the neighbours to be used

        :param y_type:
            The minority target value, just so the function can return the target
            values for the synthetic variables with correct length in a clear
            format

        :param nn_num:
            The number of nearest neighbours to be used.

        :param y_type:
            The number of synthetic samples to create.

        :param random_state:
            Seed for random number generation.

        :return:

            new: Synthetically generated samples.

            y_new: Target values for synthetic samples.
        """

        # A matrix to store the synthetic samples
        new = zeros((n_samples, len(x.T)))

        # Set seeds
        seed(random_state)
        seeds = randint(low=0, high=100*len(nn_num.flatten()), size=n_samples)

        # Randomly pick samples to construct neighbours from
        seed(random_state)
        samples = randint(low=0, high=len(nn_num.flatten()), size=n_samples)

        # Loop over the NN matrix and create new samples
        for i, n in enumerate(samples):
            # NN lines relate to original sample, columns to its nearest neighbours
            row, col = divmod(n, len(nn_num.T))

            # Take a step of random size (0,1) in the direction of the n nearest
            # neighbours
            seed(seeds[i])
            step = step_size * uniform()

            # Construct synthetic sample
            new[i] = x[row] - step * (x[row] - nn_data[nn_num[row, col]])

        # The returned target vector is simply a repetition of the minority label
        y_new = ones(len(new)) * y_type

        if verbose==True:
            print("Generated %i new samples ..." % len(new))

        return new, y_new

    @staticmethod
    def in_danger(sample, y, m, class_type, nn_obj):
        """
        Function to determine whether a given minority samples is in Danger as
        defined by Chawla, N.V et al., in: SMOTE: synthetic minority over-sampling
        technique.

        A minority sample is in danger if more than half of its nearest neighbours
        belong to the majority class. The exception being a minority sample for
        which all its nearest neighbours are from the majority class, in which case
        it is considered noise.

        :param sample:
            Sample for which danger level is to be found.

        :param y:
            Full target vector to check to which class the neighbours of sample
            belong to.

        :param m:
            The number of nearest neighbours to consider.

        :param class_type:
            The value of the target variable for the minority class.

        :param nn_obj:
            A scikit-learn NearestNeighbour object already fitted.

        :return:
            True or False depending whether a sample is in danger or not.
        """

        # Find NN for current sample
        x = nn_obj.kneighbors(sample.reshape((1, len(sample))),
                              return_distance=False)[:, 1:]

        # Count how many NN belong to the minority class
        minority = 0
        for nn in x[0]:
            if y[nn] != class_type:
                continue
            else:
                minority += 1

        # Return True of False for in danger and not in danger or noise samples.
        if minority <= m/2 or minority == m:
            # for minority == k the sample is considered to be noise and won't be
            # used, similarly to safe samples
            return False
        else:
            return True

    @staticmethod
    def is_noise(sample, y, class_type, nn_obj):
        """
        Function to determine whether a given minority sample is noise as defined
        in [1].

        A minority sample is noise if all its nearest neighbours belong to
        the majority class.

        :param sample:
            Sample for which danger level is to be found.

        :param y:
            Full target vector to check to which class the neighbours of sample
            belong to.

        :param class_type:
            The value of the target variable for the monority class.

        :param nn_obj:
            A scikit-learn NearestNeighbour object already fitted.

        :return:
            True or False depending whether a sample is in danger or not.
        """

        # Find NN for current sample
        x = nn_obj.kneighbors(sample.reshape((1, len(sample))),
                              return_distance=False)[:, 1:]

        # Check if any neighbour belong to the minority class.
        for nn in x[0]:
            if y[nn] != class_type:
                continue
            else:
                return False

        # If the loop completed, it is noise.
        return True


# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
#                       Majority under sampling children!
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
class UnderSampler(UnbalancedDataset):
    """
    Object to under sample the majority class(es) by randomly picking samples
    with or without replacement.
    """

    def __init__(self, ratio=1., random_state=None, replacement=True, verbose=True):
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
        UnbalancedDataset.__init__(self, ratio=ratio, 
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
            if (self.replacement == True):
                indx = randint(low=0, high=self.ucd[key], size=num_samples)
            else:
                indx = sample(range(np.count_nonzero(self.y == key)), num_samples)
            
            # Concatenate to the minority class
            underx = concatenate((underx, self.x[self.y == key][indx]), axis=0)
            undery = concatenate((undery, self.y[self.y == key][indx]), axis=0)

        if self.verbose==True:
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
        if self.verbose==True:
            print("Looking for majority Tomek links...")
        links = self.is_tomek(self.y, nns, self.minc, self.verbose)

        if self.verbose==True:
            print("Under-sampling performed: " + str(Counter(self.y[logical_not(links)])))

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
        
        if self.verbose==True:
            print("Under-sampling performed: " + str(Counter(undery)))

        return underx, undery

class NearMiss(UnbalancedDataset):
    """
    An implementation of NearMiss.

    See the original paper: NearMiss - "kNN Approach to Unbalanced Data Distributions: 
    A Case Study involving Information Extraction" by Zhang et al. for more details.
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
            NearMiss-3 algorithm start by a phase of resampling. This
            parameter correspond to the number of neighbours selected
            create the sub_set in which the selection will be performed.

        :param **kwargs:
            Parameter to use for the Neareast Neighbours.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, ratio=ratio, 
                                   random_state=random_state,
                                   verbose=verbose)

        # Assign the parameter of the element of this class
        ### Check that the version asked is implemented
        if not (version == 1 or version == 2 or version == 3):
            raise ValueError('UnbalancedData.NearMiss: there is only 3 versions available with parameter version=1/2/3')

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

        # Fit the minority class since that we want to know the distance to these point
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
                dist_vec, idx_vec = nn_obj.kneighbors(sub_samples_x, n_neighbors=self.size_ngh)
                # Select the right samples
                sel_x, sel_y = self.__SelectionDistBased__(dist_vec, num_samples, key, sel_strategy='nearest')
            elif self.version == 2:
                # Find the NN
                dist_vec, idx_vec = nn_obj.kneighbors(sub_samples_x, n_neighbors=self.y[self.y == self.minc].size)
                # Select the right samples
                sel_x, sel_y = self.__SelectionDistBased__(dist_vec, num_samples, key, sel_strategy='nearest')
            elif self.version == 3:
                # We need a new NN object to fit the current class
                nn_obj_cc = NearestNeighbors(n_neighbors=self.ver3_samp_ngh, **self.kwargs)
                nn_obj_cc.fit(sub_samples_x)

                # Find the set of NN to the minority class
                dist_vec, idx_vec = nn_obj_cc.kneighbors(self.x[self.y == self.minc])

                # Create the subset containing the samples found during the NN search
                ### Linearize the indexes and remove the double values
                idx_vec = np.unique(idx_vec.reshape(-1))
                ### Create the subset
                sub_samples_x = sub_samples_x[idx_vec, :]
                sub_samples_y = sub_samples_y[idx_vec]

                # Compute the NN considering the current class
                dist_vec, idx_vec = nn_obj.kneighbors(sub_samples_x, n_neighbors=self.size_ngh)
                sel_x, sel_y = self.__SelectionDistBased__(dist_vec, num_samples, key, sel_strategy='farthest')

            underx = concatenate((underx, sel_x), axis=0)
            undery = concatenate((undery, sel_y), axis=0)

        if self.verbose==True:
            print("Under-sampling performed: " + str(Counter(undery)))

        return (underx, undery)

    def __SelectionDistBased__(self, dist_vec, num_samples, key, sel_strategy='nearest'):
            
        # Compute the distance considering the farthest neighbour
        dist_avg_vec = np.sum(dist_vec[:, -self.size_ngh:], axis=1)
            
        # Sort the list of distance and get the index
        if sel_strategy == 'nearest':
            sort_way = False
        elif sel_strategy == 'farthest':
            sort_way = True
        else:
            raise ValueError('Unbalanced.NearMiss: the sorting can be done only with nearest or farthest data points.')

        sorted_idx = sorted(range(len(dist_avg_vec)), key=dist_avg_vec.__getitem__, reverse=sort_way)

        # Select the desired number of samples
        sel_idx = sorted_idx[:num_samples]

        return (self.x[self.y == key][sel_idx], self.y[self.y == key][sel_idx])

class CondensedNearestNeighbour(UnbalancedDataset):
    """
    An implementation of Condensend Neareat Neighbour.

    See the original paper: CNN - "Addressing the Curse of Imbalanced Training Set: One-Sided Selection" by Khubat et al. for more details.
    """

    def __init__(self, ratio=1., random_state=None, 
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
        UnbalancedDataset.__init__(self, ratio=ratio, 
                                   random_state=random_state,
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
            maj_sample = sample(self.x[self.y == key], self.n_seeds_S)
            
            # Create the set C
            C_x = np.append(self.x[self.y == self.minc], maj_sample, axis=0)
            C_y = np.append(self.y[self.y == self.minc], [key] * self.n_seeds_S)

            # Create the set S
            S_x = self.x[self.y == key]
            S_y = self.y[self.y == key]
            
            # Create a k-NN classifier
            knn = KNeighborsClassifier(n_neighbors=self.size_ngh, **self.kwargs)
            
            # Fit C into the knn
            knn.fit(C_x, C_y)

            # Classify on S
            pred_S_y = knn.predict(S_x)

            # Find the misclassified S_y
            sel_x = np.squeeze(S_x[np.nonzero(pred_S_y != S_y), :])
            sel_y = S_y[np.nonzero(pred_S_y != S_y)]

            underx = concatenate((underx, sel_x), axis=0)
            undery = concatenate((undery, sel_y), axis=0)

        if self.verbose==True:
            print("Under-sampling performed: " + str(Counter(undery)))

        return (underx, undery)

class OneSidedSelection(UnbalancedDataset):
    """
    An implementation of One-Sided Selection.

    See the original paper: OSS - "Addressing the Curse of Imbalanced Training Set: One-Sided Selection" by Khubat et al. for more details.
    """

    def __init__(self, ratio=1., random_state=None, 
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
        UnbalancedDataset.__init__(self, ratio=ratio, 
                                   random_state=random_state,
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
            maj_sample = sample(self.x[self.y == key], self.n_seeds_S)
            
            # Create the set C
            C_x = np.append(self.x[self.y == self.minc], maj_sample, axis=0)
            C_y = np.append(self.y[self.y == self.minc], [key] * self.n_seeds_S)

            # Create the set S
            S_x = self.x[self.y == key]
            S_y = self.y[self.y == key]
            
            # Create a k-NN classifier
            knn = KNeighborsClassifier(n_neighbors=self.size_ngh, **self.kwargs)
            
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
        if self.verbose==True:
            print("Looking for majority Tomek links...")
        links = self.is_tomek(undery, nns, self.minc, self.verbose)

        if self.verbose==True:
            print("Under-sampling performed: " + str(Counter(undery[logical_not(links)])))

        # Return data set without majority Tomek links.
        return underx[logical_not(links)], undery[logical_not(links)]


class NeighboorhoodCleaningRule(UnbalancedDataset):
    """
    An implementation of Neighboorhood Cleaning Rule.

    See the original paper: NCL - "Improving identification of difficult small classes by balancing class distribution" by Laurikkala et al. for more details.
    """

    def __init__(self, ratio=1., random_state=None, 
                 size_ngh=3, verbose=True, **kwargs):
        """
        :param size_ngh
            Size of the neighbourhood to consider in order to make
            the comparison between each samples and their NN.
        
        :param **kwargs
            Parameter to use for the Neareast Neighbours.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, ratio=ratio, 
                                   random_state=random_state,
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
            if (key == self.minc):
                # Get the index to exclude
                idx_to_exclude = idx_to_exclude + nnhood_idx[np.nonzero(nnhood_label[np.nonzero(nnhood_bool)])].tolist()
            else:
                # Get the index to exclude
                idx_to_exclude = idx_to_exclude +idx_sub_sample[np.nonzero(nnhood_bool)].tolist()

        # Create a vector with the sample to select
        sel_idx = np.ones(self.y.shape)
        sel_idx[idx_to_exclude] = 0

        # Get the samples from the majority classes
        sel_x = np.squeeze(self.x[np.nonzero(sel_idx), :])
        sel_y = self.y[np.nonzero(sel_idx)]
        
        underx = concatenate((underx, sel_x), axis=0)
        undery = concatenate((undery, sel_y), axis=0)

        if self.verbose==True:
            print("Under-sampling performed: " + str(Counter(undery)))

        return (underx, undery)

# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
#                      Minority over sampling children!
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
class OverSampler(UnbalancedDataset):
    """
    Object to over-sample the minority class(es) by picking samples at random
    with replacement.

    *Supports multiple classes.
    """

    def __init__(self, ratio=1., random_state=None, verbose=True):
        """
        :param ratio:
            Number of samples to draw with respect to the number of samples in
            the original minority class.
                N_new =

        :param random_state:
            Seed.

        :return:
            Nothing.
        """
        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state,
                                   verbose=verbose)

    def resample(self):
        """
        Over samples the minority class by randomly picking samples with
        replacement.

        :return:
            overx, overy: The features and target values of the over-sampled
            data set.
        """

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

        if self.verbose==True:
            print("Over-sampling performed: " + str(Counter(overy)))

        # Return over sampled dataset
        return overx, overy


class SMOTE(UnbalancedDataset):
    """
    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique, and the variations Borderline SMOTE 1, 2 and
    SVM-SMOTE.

    See the original papers: [1], [2], [3] for more details.

    * It does not support multiple classes automatically, but can be called
    multiple times
    """

    def __init__(self, k=5, m=10, out_step=0.5, ratio=1, random_state=None,
                 kind='regular', verbose=0,
                 **kwargs):
        """
        SMOTE over sampling algorithm and variations. Choose one of the
        following options: 'regular', 'borderline1', 'borderline2', 'svm'

        :param k: Number of nearest neighbours to used to construct synthetic
                  samples.

        :param m: The number of nearest neighbours to use to determine if a
                  minority sample is in danger.

        :param out_step: Step size when extrapolating

        :param ratio: Fraction of the number of minority samples to
                      synthetically generate.

        :param random_state: Seed for random number generation

        :param kind: The type of smote algorithm to use one of the following
                     options: 'regular', 'borderline1', 'borderline2', 'svm'

        :param verbose: Whether or not to print status information

        :param kwargs: Additional arguments passed to sklearn SVC object
        """

        # Parent class methods
        UnbalancedDataset.__init__(self,
                                   ratio=ratio,
                                   random_state=random_state)

        # --- The type of smote
        # This object can perform regular smote over-sampling, borderline 1,
        # borderline 2 and svm smote. Since the algorithms are fairly simple
        # they share most methods.#
        self.kind = kind

        # --- Verbose
        # Control whether or not status and progress information should be#
        self.verbose = verbose

        # --- Nearest Neighbours for synthetic samples
        # The smote algorithm uses the k-th nearest neighbours of a minority
        # sample to generate new synthetic samples.#
        self.k = k

        # --- NN object
        # Import the NN object from scikit-learn library. Since in the smote
        # variations we must first find samples that are in danger, we
        # initialize the NN object differently depending on the method chosen#
        from sklearn.neighbors import NearestNeighbors

        if kind == 'regular':
            # Regular smote does not look for samples in danger, instead it
            # creates synthetic samples directly from the k-th nearest
            # neighbours with not filtering#
            self.nearest_neighbour_ = NearestNeighbors(n_neighbors=k + 1)
        else:
            # Borderline1, 2 and SVM variations of smote must first look for
            # samples that could be considered noise and samples that live
            # near the boundary between the classes. Therefore, before
            # creating synthetic samples from the k-th nns, it first look
            # for m nearest neighbors to decide whether or not a sample is
            # noise or near the boundary.#
            self.nearest_neighbour_ = NearestNeighbors(n_neighbors=m + 1)

            # --- Nearest Neighbours for noise and boundary (in danger)
            # Before creating synthetic samples we must first decide if
            # a given entry is noise or in danger. We use m nns in this step#
            self.m = m

        # --- SVM smote
        # Unlike the borderline variations, the SVM variation uses the support
        # vectors to decide which samples are in danger (near the boundary).
        # Additionally it also introduces extrapolation for samples that are
        # considered safe (far from boundary) and interpolation for samples
        # in danger (near the boundary). The level of extrapolation is
        # controled by the out_step.#
        if kind == 'svm':
            # As usual, use scikit-learn object#
            from sklearn.svm import SVC

            # Store extrapolation size#
            self.out_step = out_step

            # Store SVM object with any parameters#
            self.svm_ = SVC(**kwargs)

    def resample(self):
        """
        Main method of all children classes.

        :return: Over-sampled data set.
        """

        # Start by separating minority class features and target values.
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # If regular SMOTE is to be performed#
        if self.kind == 'regular':
            # Print if verbose is true#
            if self.verbose:
                print("Finding the %i nearest neighbours..." % self.k, end="")

            # Look for k-th nearest neighbours, excluding, of course, the
            # point itself.#
            self.nearest_neighbour_.fit(minx)

            # Matrix with k-th nearest neighbours indexes for each minority
            # element.#
            nns = self.nearest_neighbour_.kneighbors(minx,
                                                     return_distance=False)[:, 1:]

            # Print status if verbose is true#
            if self.verbose:
                ##
                print("done!")

                # Creating synthetic samples #
                print("Creating synthetic samples...", end="")

            # --- Generating synthetic samples
            # Use static method make_samples to generate minority samples
            # FIX THIS SHIT!!!#
            sx, sy = self.make_samples(x=minx,
                                       nn_data=minx,
                                       y_type=self.minc,
                                       nn_num=nns,
                                       n_samples=int(self.ratio * len(miny)),
                                       step_size=1.0,
                                       random_state=self.rs)

            if self.verbose:
                print("done!")

            # Concatenate the newly generated samples to the original data set
            ret_x = concatenate((self.x, sx), axis=0)
            ret_y = concatenate((self.y, sy), axis=0)

            return ret_x, ret_y

        if (self.kind == 'borderline1') or (self.kind == 'borderline2'):

            if self.verbose:
                print("Finding the %i nearest neighbours..." % self.m, end="")

            # Find the NNs for all samples in the data set.
            self.nearest_neighbour_.fit(self.x)

            if self.verbose:
                print("done!")

            # Boolean array with True for minority samples in danger
            danger_index = [self.in_danger(x, self.y, self.m, miny[0],
                            self.nearest_neighbour_) for x in minx]

            # Turn into numpy array#
            danger_index = asarray(danger_index)

            # If all minority samples are safe, return the original data set.
            if not any(danger_index):
                ##
                if self.verbose:
                    print('There are no samples in danger. No borderline '
                          'synthetic samples created.')

                # All are safe, nothing to be done here.#
                return self.x, self.y

            # If we got here is because some samples are in danger, we need to
            # find the NNs among the minority class to create the new synthetic
            # samples.
            #
            # We start by changing the number of NNs to consider from m + 1
            # to k + 1
            self.nearest_neighbour_.set_params(**{'n_neighbors': self.k + 1})
            self.nearest_neighbour_.fit(minx)

            # nns...#
            nns = self.nearest_neighbour_.kneighbors(minx[danger_index],
                                                     return_distance=False)[:, 1:]

            # B1 and B2 types diverge here!!!
            if self.kind == 'borderline1':
                # Create synthetic samples for borderline points.
                sx, sy = self.make_samples(minx[danger_index], minx, miny[0], nns,
                                           int(self.ratio * len(miny)),
                                           random_state=self.rs)

                # Concatenate the newly generated samples to the original data set
                ret_x = concatenate((self.x, sx), axis=0)
                ret_y = concatenate((self.y, sy), axis=0)

                return ret_x, ret_y

            else:
                # Split the number of synthetic samples between only minority
                # (type 1), or minority and majority (with reduced step size)
                # (type 2).
                np.random.seed(self.rs)

                # The fraction is sampled from a beta distribution centered
                # around 0.5 with variance ~0.01#
                fractions = betavariate(alpha=10, beta=10)

                # Only minority
                sx1, sy1 = self.make_samples(minx[danger_index], minx, self.minc, nns,
                                             fractions * (int(self.ratio * len(miny)) + 1),
                                             step_size=1,
                                             random_state=self.rs)

                # Only majority with smaller step size
                sx2, sy2 = self.make_samples(minx[danger_index], self.x[self.y != self.minc],
                                             self.minc, nns,
                                             (1 - fractions) * int(self.ratio * len(miny)),
                                             step_size=0.5,
                                             random_state=self.rs)

                # Concatenate the newly generated samples to the original data set
                ret_x = np.concatenate((self.x, sx1, sx2), axis=0)
                ret_y = np.concatenate((self.y, sy1, sy2), axis=0)

                return ret_x, ret_y

        if self.kind == 'svm':
            # The SVM smote model fits a support vector machine
            # classifier to the data and uses the support vector to
            # provide a notion of boundary. Unlike regular smote, where
            # such notion relies on proportion of nearest neighbours
            # belonging to each class.#

            # Fit SVM to the full data#
            self.svm_.fit(self.x, self.y)

            # Find the support vectors and their corresponding indexes
            support_index = self.svm_.support_[self.y[self.svm_.support_] == self.minc]
            support_vector = self.x[support_index]

            # First, find the nn of all the samples to identify samples in danger
            # and noisy ones
            if self.verbose:
                print("Finding the %i nearest neighbours..." % self.m, end="")

            # As usual, fit a nearest neighbour model to the data
            self.nearest_neighbour_.fit(self.x)

            if self.verbose:
                print("done!")

            # Now, get rid of noisy support vectors

            # Boolean array with True for noisy support vectors
            noise_bool = []
            for x in support_vector:
                noise_bool.append(self.is_noise(x, self.y, self.minc,
                                                self.nearest_neighbour_))

            # Turn into array#
            noise_bool = asarray(noise_bool)

            # Remove noisy support vectors
            support_vector = support_vector[np.logical_not(noise_bool)]

            # Find support_vectors there are in danger (interpolation) or not
            # (extrapolation)
            danger_bool = [self.in_danger(x, self.y, self.m, self.minc,
                                          self.nearest_neighbour_)
                           for x in support_vector]

            # Turn into array#
            danger_bool = asarray(danger_bool)

            #Something ...#
            safety_bool = np.logical_not(danger_bool)

            #things to print#
            print_stats = (len(support_vector),
                           noise_bool.sum(),
                           danger_bool.sum(),
                           safety_bool.sum())

            if self.verbose:
                print("Out of %i support vectors, %i are noisy, %i are in danger "
                      "and %i are safe." % print_stats)

                # Proceed to find support vectors NNs among the minority class
                print("Finding the %i nearest neighbours..." % self.k, end="")

            self.nearest_neighbour_.set_params(**{'n_neighbors': self.k + 1})
            self.nearest_neighbour_.fit(minx)

            if self.verbose:
                print("done!")
                print("Creating synthetic samples...", end="")

            # Split the number of synthetic samples between interpolation and
            # extrapolation

            # The fraction are sampled from a beta distribution with mean
            # 0.5 and variance 0.01#
            np.random.seed(self.rs)
            fractions = betavariate(alpha=10, beta=10)

            # Interpolate samples in danger
            nns = self.nearest_neighbour_.kneighbors(support_vector[danger_bool],
                                                     return_distance=False)[:, 1:]

            sx1, sy1 = self.make_samples(support_vector[danger_bool], minx,
                                         self.minc, nns,
                                         fractions * (int(self.ratio * len(minx)) + 1),
                                         step_size=1,
                                         random_state=self.rs)

            # Extrapolate safe samples
            nns = self.nearest_neighbour_.kneighbors(support_vector[safety_bool],
                                                     return_distance=False)[:, 1:]

            sx2, sy2 = self.make_samples(support_vector[safety_bool], minx,
                                         self.minc, nns,
                                         (1 - fractions) * int(self.ratio * len(minx)),
                                         step_size=-self.out_step,
                                         random_state=self.rs)

            if self.verbose:
                print("done!")

            # Concatenate the newly generated samples to the original data set
            ret_x = concatenate((self.x, sx1, sx2), axis=0)
            ret_y = concatenate((self.y, sy1, sy2), axis=0)

            return ret_x, ret_y


class SMOTETomek(UnbalancedDataset):
    """
    An implementation of SMOTE + Tomek.

    Comparison performed in "Balancing training data for automated annotation of keywords: 
    a case study", Batista et al. for more details.
    """

    def __init__(self, k=5, ratio=1., random_state=None, verbose=True):
        """
        :param k:
            Number of nearest neighbours to use when constructing the synthetic
            samples.

        :param ratio:
            Fraction of the number of minority samples to synthetically
            generate.

        :param random_state:
            Seed.

        :return:
            The resampled data set with synthetic samples concatenated at the
            end.
        """

        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state,
                                   verbose=verbose)

        # Instance variable to store the number of neighbours to use.
        self.k = k

    def resample(self):
        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Finding nns
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
        
        from sklearn.neighbors import NearestNeighbors

        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(ret_x)
        nns = nn.kneighbors(ret_x, return_distance=False)[:, 1]

        # Send the information to is_tomek function to get boolean vector back
        links = self.is_tomek(ret_y, nns, self.minc, self.verbose)

        if self.verbose==True:
            print("Over-sampling performed: " + str(Counter(ret_y[logical_not(links)])))

        # Return data set without majority Tomek links.
        return ret_x[logical_not(links)], ret_y[logical_not(links)]

class SMOTEENN(UnbalancedDataset):
    """
    An implementation of SMOTE + ENN.

    Comparison performed in "A study of the behavior of several methods for balancing
    machine learning training data", Batista et al. for more details.

    """

    def __init__(self, k=5, ratio=1., random_state=None, 
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
            Fraction of the number of minority samples to synthetically
            generate.

        :param random_state:
            Seed.

        :return:
            The resampled data set with synthetic samples concatenated at the
            end.
        """

        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state,
                                   verbose=verbose)

        # Instance variable to store the number of neighbours to use.
        self.k = k
        self.size_ngh = size_ngh
        self.kwargs = kwargs

    def resample(self):
        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Finding nns
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

        # Import the k-NN classifier
        from sklearn.neighbors import NearestNeighbors

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
            nnhood_idx = nn_obj.kneighbors(sub_samples_x, return_distance=False)
                                
            # Get the label of the corresponding to the index
            nnhood_label = (ret_y[nnhood_idx] == key)
                
            # Check which one are the same label than the current class
            # Make an AND operation through the k neighbours
            nnhood_bool = np.all(nnhood_label, axis=1)

            # Get the samples which agree all together
            sel_x = np.squeeze(sub_samples_x[np.nonzero(nnhood_bool), :])
            sel_y = sub_samples_y[np.nonzero(nnhood_bool)]
        
            if (key_idx == 0):
                underx = sel_x[:, :]
                undery = sel_y[:]
            else:
                underx = concatenate((underx, sel_x), axis=0)
                undery = concatenate((undery, sel_y), axis=0)

        if self.verbose==True:
            print("Over-sampling performed: " + str(Counter(undery)))

        return (underx, undery)

# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
#                      Ensemble Set by Under-Sampling!
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #

class EasyEnsemble(UnderSampler):
    """
    Object to perform classification on balanced ensembled selected from 
    random sampling.

    It is based on the idea presented in the paper "Exploratory Undersampling
    Class-Imbalance Learning" by Liu et al.
    """

    def __init__(self, ratio=1., random_state=None, replacement=False, 
                 n_subsets=10, verbose=True):
        """
        :param ratio:
            The ratio of majority elements to sample with respect to the number
            of minority cases.

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
            Python list contatining the different data arrays generated and balanced.

        :return subsets_y:
            Python list contraining the different label arrays generated and balanced.
        """

        subsets_x = []
        subsets_y = []

        for s in range(self.n_subsets):
            if self.verbose==True:
                print("Creation of the set #%i" % s)
            tmp_subset_x, tmp_subset_y = UnderSampler.resample(self)
            subsets_x.append(tmp_subset_x)
            subsets_y.append(tmp_subset_y)

        return subsets_x, subsets_y

class BalanceCascade(UnbalancedDataset):
    """
    Object to perform classification on balanced ensembled selected from 
    random sampling and selected using classifier.

    It is based on the idea presented in the paper "Exploratory Undersampling
    Class-Imbalance Learning" by Liu et al.
    """

    def __init__(self, ratio=1., random_state=None, n_max_subset=None, 
                 classifier='knn', bootstrap=True, verbose=True, **kwargs):
        """
        :param ratio:
            The ratio of majority elements to sample with respect to the number
            of minority cases.

        :param random_state:
            Seed.
        
        :param n_max_subset:
            Maximum number of subsets to generate. By default, all data from the
            training will be selected that could lead to a large number of subsets.
            We can probably reduced this number empirically.

        :param classifier:
            The classifier that will be selected to confront the prediction
            with the real labels.

        :param **kwargs:
            The parameters associated with the classifier provided.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, ratio=ratio, 
                                   random_state=random_state, 
                                   verbose=verbose)

        # Define the classifier to use
        if (classifier == 'knn'):
            from sklearn.neighbors import KNeighborsClassifier
            self.classifier = KNeighborsClassifier(**kwargs)
        elif (classifier == 'decision-tree'):
            from sklearn.tree import DecisionTreeClassifier
            self.classifier = DecisionTreeClassifier(**kwargs)
        elif (classifier == 'random-forest'):
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(**kwargs)
        elif (classifier == 'adaboost'):
            from sklearn.ensemble import AdaBoostClassifier
            self.classifier = AdaBoostClassifier(**kwargs)
        elif (classifier == 'gradient-boosting'):
            from sklearn.ensemble import GradientBoostingClassifier
            self.classifier = GradientBoostingClassifier(**kwargs)
        elif (classifier == 'linear-svm'):
            from sklearn.svm import LinearSVC
            self.classifier = LinearSVC(**kwargs)
        else:
            raise ValueError('UnbalancedData.BalanceCascade: classifier not yet supported.')

        self.n_max_subset = n_max_subset
        self.classifier_name = classifier
        self.bootstrap = bootstrap

    def resample(self):
        """
        :return subsets_x:
            Python list contatining the different data arrays generated and balanced.

        :return subsets_y:
            Python list contraining the different label arrays generated and balanced.
        """

        subsets_x = []
        subsets_y = []

        # Start with the minority class
        min_x = self.x[self.y == self.minc]
        min_y = self.y[self.y == self.minc]

        # Condition to initiliase before the search
        b_subset_search = True
        n_subsets = 0
        # Get the initial number of samples to select in the majority class
        n_elt_maj = self.ucd[self.minc]
        # Create the array characterising the array containing the majority class
        N_x = self.x[self.y != self.minc]
        N_y = self.y[self.y != self.minc]
        b_sel_N = np.array([True] * N_y.size)
        idx_mis_class = np.array([])
        
        # Loop to create the different subsets
        while b_subset_search:
            # Generate an appropriate number of index to extract
            # from the majority class depending of the false classification
            # rate of the previous iteration
            idx_sel_from_maj = np.array(sample(np.nonzero(b_sel_N)[0], n_elt_maj))
            idx_sel_from_maj = np.concatenate((idx_mis_class, idx_sel_from_maj), axis=0).astype(int)
           
            # Mark these indexes as not being considered for next sampling
            b_sel_N[idx_sel_from_maj] = False

            # For now, we will train and classify on the same data
            # Let see if we should find another solution. Anyway,
            # random stuff are still random stuff
            x_data = concatenate((min_x, N_x[idx_sel_from_maj, :]), axis=0)
            y_data = concatenate((min_y, N_y[idx_sel_from_maj]), axis=0)

            # Push these data into a new subset
            subsets_x.append(x_data)
            subsets_y.append(y_data)

            if (not ( (self.classifier_name == 'knn'       ) or 
                      (self.classifier_name == 'linear-svm')   )
                and self.bootstrap):
                # Apply a bootstrap on x_data
                curr_sample_weight = np.ones((y_data.size,), dtype=np.float64)
                indices = np.random.randint(0, y_data.size, y_data.size)
                sample_counts = np.bincount(indices, minlength=y_data.size)
                curr_sample_weight *= sample_counts
            
                # Train the classifier using the current data
                self.classifier.fit(x_data, y_data, curr_sample_weight)
                
            else:
                # Train the classifier using the current data
                self.classifier.fit(x_data, y_data)
                

            # Predict using only the majority class
            pred_label = self.classifier.predict(N_x[idx_sel_from_maj, :])

            # Basically let's find which sample have to be retained for the
            # next round

            # Find the misclassified index to keep them for the next round
            idx_mis_class = idx_sel_from_maj[np.nonzero(pred_label != N_y[idx_sel_from_maj])]
            # Count how many random element will be selected
            n_elt_maj = self.ucd[self.minc] - idx_mis_class.size
            
            if self.verbose==True:
                print("Creation of the subset #" + str(n_subsets))

            # We found a new subset, increase the counter
            n_subsets = n_subsets + 1
                        
            # Check if we have to make an early stopping
            if (self.n_max_subset != None):
                if (self.n_max_subset >= n_subsets):
                    b_subset_search = False
                    if self.verbose==True:
                        print('The number of subset achieved their maximum')

            # Also check that we will have enough sample to extract at the 
            # next round
            if (n_elt_maj > np.count_nonzero(b_sel_N)):
                b_subset_search = False
                # Select the remaining data
                idx_sel_from_maj = np.nonzero(b_sel_N)[0]
                idx_sel_from_maj = np.concatenate((idx_mis_class, idx_sel_from_maj), axis=0).astype(int)
                # Select the final batch
                x_data = concatenate((min_x, N_x[idx_sel_from_maj, :]), axis=0)
                y_data = concatenate((min_y, N_y[idx_sel_from_maj]), axis=0)
                # Push these data into a new subset
                subsets_x.append(x_data)
                subsets_y.append(y_data)
                if self.verbose==True:
                    print("Creation of the subset #" + str(n_subsets))

                # We found a new subset, increase the counter
                n_subsets = n_subsets + 1

                if self.verbose==True:
                    print('Not enough samples to continue creating subsets')

       
        # Return the different subsets
        return subsets_x, subsets_y


# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
#                                  Pipeline!!!
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
# ----------------------------------- // ----------------------------------- #
class Pipeline:
    """
    A helper object to concatenate a number of re sampling objects and
    streamline the re-sampling process.
    """
    
    def __init__(self, x, y):
        """
        :param x:
            Feature matrix.

        :param y:
            Target vectors.
        """
        
        self.x = x
        self.y = y

    def pipeline(self, list_methods):
        """
        :param list_methods:
            Pass the methods to be used in a list, in the order they will be
            used.

        :return:
            The re-sampled dataset.
        """

        # Initialize with the original dataset.
        x, y = self.x, self.y

        # Go through the list of methods and fit_transform each to the result
        # of the last.
        for met in list_methods:
            x, y = met.fit_transform(x, y)
            print(x.shape)

        # Return the re-sampled dataset.
        return x, y
