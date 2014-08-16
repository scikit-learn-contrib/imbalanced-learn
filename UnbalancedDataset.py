'''
UnbalancedDataset
=================

UnbalancedDataset is a python module offering a number of resampling techniques commonly used in datasets showing strong between-class imbalance.

Most classification algorithms will only perform optimally when the number of samples of each class is roughly the same. Highly skewed datasets, where the minority heavily outnumbered by one or more classes, haven proven to be a challenge while at the same time becoming more and more common.

One way of addresing this issue is by resampling the dataset as to offset this imbalance with the hope of arriving and a more robust and fair decision boundary than you would otherwise.

Resampling techniques are divided in two categories:
    1. Under-sampling the majority class(es).
    2. Over-sampling the minority class.

Bellow is a list of the methods currently implemented in this module.

* Under-sampling
    1. Random majority under-sampling with replacement
    2. Extraction of majority-minority Tomek links
    3. Under-sampling with Cluster Centroids

* Over-sampling
    1. Random minority over-sampling with replacement
    2. SMOTE - Synthetic Minority Over-sampling Technique
    3. bSMOTE(1&2) - Borderline SMOTE of types 1 and 2
    4. SVM_SMOTE - Support Vectors SMOTE

This is a work in progress. Any comments, suggestions or corrections are welcome.

References:

SMOTE - "SMOTE: synthetic minority over-sampling technique" by Chawla, N.V et al.

Borderline SMOTE -  "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning, Hui Han, Wen-Yuan Wang, Bing-Huan Mao"

SVM_SMOTE - "Borderline Over-sampling for Imbalanced Data Classification, Nguyen, Cooper, Kamei"


TO DO LIST:
===========
    1. Add control for level of verbosity.
'''



from __future__ import division
from __future__ import print_function

__author__ = 'fnogueira'


#import numpy

from random import gauss
from random import seed as Pyseed

from numpy.random import seed, randint, uniform
from numpy import zeros, ones, concatenate, logical_not, asarray
from numpy import sum as nsum

# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
#                                            Functions!!!
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
def is_tomek(y, nns, class_type):
    '''
    is_tomek uses the target vector and the first neighbour of every sample point and looks for Tomek
    pairs. Returning a boolean vector with True for majority Tomek links.

    :param y:
        Target vector of the data set, necessary to keep track of whether a sample belongs to minority or not

    :param nns:
        The index of the closes nearest neighbour to a sample point.

    :param class_type:
        The label of the minority class.

    :return:
        Boolean vector on len( # samples ), with True for majority samples that are Tomek links.
    '''

    # Initialize the boolean result as false, and also a counter
    links = zeros(len(y), dtype=bool)
    count = 0

    # Loop through each sample and looks whether it belongs to the minority class. If it does, we don't consider it
    # since we want to keep all minority samples. If, however, it belongs to the majority sample we look at its first
    # neighbour. If its closest neighbour also has the current sample as its closest neighbour, the two form a Tomek
    # link.
    for ind, ele in enumerate(y):

        if ele == class_type:
            continue

        if y[nns[ind]] == class_type:

            # If they form a tomek link, put a True marker on this sample, and increase counter by one.
            if nns[nns[ind]] == ind:
                links[ind] = True
                count += 1

    print("%i Tomek links found." % count)

    return links


# -------------------------------- // -------------------------------- #
def make_samples(x, X, ytype, nns, nsamples, step_size = 1, random_state = None):
    '''
    A support function that returns artificial samples constructed along the line connecting nearest neighbours.

    :param x:
        Minority points for which new samples are going to be created.

    :param X:
        Data set carrying all the neighbours to be used

    :param ytype:
        The minority target value, just so the function can return the target values for the
        synthetic variables with correct length in a clear format

    :param nns:
        The number of nearest neighbours to be used.

    :param ytype:
        The number of synthetic samples to create.

    :param random_state:
        Seed for random number generation.

    :return:

        new: Syntheticaly generated samples.

        y_new: Target values for synthetic samples.
    '''

    # A matrix to store the synthetic samples
    new = zeros((nsamples, len(x.T)))

    # Set seeds
    seed(random_state)
    seeds = randint(low=0, high=100*len(nns.flatten()), size=nsamples)

    # Randomly pick samples to construct neighbours from
    seed(random_state)
    samples = randint(low=0, high=len(nns.flatten()), size=nsamples)

    # Loop over the NN matrix and create new samples
    for i, n in enumerate(samples):
        # NN lines relate to original sample, columns to its nearest neighbours
        row, col = divmod(n, len(nns.T))

        # Take a step of random size (0,1) in the direction of the n nearest neighbours
        seed(seeds[i])
        step = step_size * uniform()

        # Construct synthetic sample
        new[i] = x[row] - step * (x[row] - X[nns[row, col]])

    # The returned target vector is simply a repetition of the minority label
    y_new = ones(len(new)) * ytype

    return new, y_new

# -------------------------------- // -------------------------------- #
def in_danger(sample, Y, m, class_type, nn_obj):
    '''
    Function to determine whether a given minority samples is in Danger as defined by
    Chawla, N.V et al., in: SMOTE: synthetic minority over-sampling technique.

    A minority sample is in danger if more than half of its nearest neighbours belong to
    the majority class. The exception being a minority sample for which all its nearest neighbours are
    from the majority class, in which case it is considered noise.

    :param sample:
        Sample for which danger level is to be found.

    :param Y:
        Full target vector to check to which class the neighbours of sample belong to.

    :param m:
        The number of nearest neighbours to consider.

    :param class_type:
        The value of the target variable for the monority class.

    :param nn_obj:
        A scikit-learn NearestNeighbour object already fitted.

    :return:
        True or False depending whether a sample is in danger or not.
    '''

    # Find NN for current sample
    x = nn_obj.kneighbors(sample.reshape((1, len(sample))), return_distance=False)[:, 1:]

    # Count how many NN belong to the minority class
    minority = 0
    for nn in x[0]:
        if Y[nn] != class_type:
            continue
        else:
            minority += 1

    # Return True of False for in danger and not in danger or noise samples.
    if minority <= m/2 or minority == m:
        # for minority == k the sample is considered to be noise and won't be used, similarly to safe samples
        return False
    else:
        return True


# -------------------------------- // -------------------------------- #
def is_noise(sample, Y, m, class_type, nn_obj):
    '''
    Function to determine whether a given minority samples is noise as defined by
    Chawla, N.V et al., in: SMOTE: synthetic minority over-sampling technique.

    A minority sample is noise if all its nearest neighbours belong to
    the majority class.

    :param sample:
        Sample for which danger level is to be found.

    :param Y:
        Full target vector to check to which class the neighbours of sample belong to.

    :param m:
        The number of nearest neighbours to consider.

    :param class_type:
        The value of the target variable for the monority class.

    :param nn_obj:
        A scikit-learn NearestNeighbour object already fitted.

    :return:
        True or False depending whether a sample is in danger or not.
    '''

    # Find NN for current sample
    x = nn_obj.kneighbors(sample.reshape((1, len(sample))), return_distance=False)[:, 1:]

    # Check if any neighbour belong to the minority class.
    for nn in x[0]:
        if Y[nn] != class_type:
            continue
        else:
            return False

    # If the loop completed, it is noise.
    return True


# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
#                                           Parent Class!
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
class UnbalancedDataset:
    '''
    Parent class with the main methods: fit, transform and fit_transform
    '''

    def __init__(self, ratio=1, random_state=None):
        '''
        Initialize this object and its instance variables.

        :param ratio:
            ratio will be used in different ways for different children object. But in general it quantifies
            the amount of under sampling or over sampling to be perfomed with respect to the number of
            samples present in the minority class.

        :param random_state:
            Seed for random number generation.

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
            Dictionary to hold the label of all the class and the number of elements in each.
            {'label A' : #a, 'label B' : #b, ...}
        '''

        self.ratio = ratio
        self.rs = random_state

        self.x = None
        self.y = None

        self.minc = None
        self.maxc = None
        self.ucd = {}

    def fit(self, X, y):
        '''
        Class method to find the relevant class statistics and store it.

        :param X:
            Features.

        :param y:
            Target values.

        :return:
            Nothing
        '''

        self.x = X
        self.y = y

        print("Determining class statistics...", end = "")

        # Get all the unique elements in the target array
        uniques = set(self.y)
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

        for key in self.ucd.keys():

            if self.ucd[key] < curre_min:
                self.minc = key
                curre_min = self.ucd[key]

            if self.ucd[key] > curre_max:
                self.maxc = key
                curre_max = self.ucd[key]
        print("done!")


    def transform(self):
        '''
        Class method to resample the dataset with a particular technique.

        :return:
            The resampled data set.
        '''

        self.out_x, self.out_y = self.resample()

        return self.out_x, self.out_y

    def fit_transform(self, X, y):
        '''
        Class method to fit and transform the data set automatically.

        :param X:
            Features.

        :param y:
            Target values.

        :return:
            The resampled data set.
        '''

        self.fit(X, y)
        self.out_x, self.out_y = self.resample()

        return self.out_x, self.out_y

# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
#                                 Majority under sampling children!
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
class UnderSampler(UnbalancedDataset):
    '''
    Object to under sample the majority class(es) by randomly picking samples with replacement.
    '''

    def __init__(self, ratio = 1, random_state = None):
        '''
        :param ratio:
            The ratio of majority elements to sample with respect to the number of minority cases.

        :param random_state:
            Seed.

        :return:
            underx, undery: The features and target values of the under-sampled data set.
        '''
        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

    def resample(self):
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
            indx = randint(low = 0, high = self.ucd[key], size = num_samples)

            # Concatenate to the minority class
            underx = concatenate((underx, self.x[self.y == key][indx]), axis = 0)
            undery = concatenate((undery, self.y[self.y == key][indx]), axis = 0)

        return underx, undery


# -------------------------------- // -------------------------------- #
class TomekLinks(UnbalancedDataset):
    '''
    Object to identify and remove majority samples that form a Tomek link with minority samples.
    '''

    def __init__(self):
        '''
        No parameters.

        :return:
            Nothing.
        '''
        UnbalancedDataset.__init__(self)

    def resample(self):
        '''
        :return:
            Return the data with majority samples that form a Tomek link removed.
        '''

        from sklearn.neighbors import NearestNeighbors

        # Find the nearest neighbour of every point
        print("Finding nearest neighbour...", end="")
        NN = NearestNeighbors(n_neighbors = 2)
        NN.fit(self.x)
        nns = NN.kneighbors(self.x, return_distance=False)[:, 1]
        print("done!")

        # Send the information to is_tomek function to get boolean vector back
        print("Looking for majority Tomek links...", end="")
        links = is_tomek(self.y, nns, self.minc)

        # Return data set without majority Tomek links.
        return self.x[logical_not(links)], self.y[logical_not(links)]


# -------------------------------- // -------------------------------- #
# Experimental techniques
class ClusterCentroids(UnbalancedDataset):
    '''
    Experimental method that under samples the majority class by replacing a cluster of
    majority samples by the cluster centroid of a KMeans algorithm.

    This algorithm keeps N majority samples by fitting the KMeans algorithm with N cluster
    to the majority class and using the coordinates of the N cluster centroids as the new
    majority samples.
    '''

    def __init__(self, kargs = {}, ratio = 1, random_state = None):
        '''
        :param kargs:
            Arguments the user might want to pass to the KMeans object from scikit-learn.

        :param ratio:
            The number of cluster to fit with respect to the number of samples in the minority class.
            N_clusters = int( ratio * N_minority_samples) = N_majority_undersampled.

        :param random_state:
            Seed.

        :return:
            Under sampled data set.
        '''
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

        self.kargs = kargs

    def resample(self):
        '''


        :param ratio:
            The ratio of number of majority cluster centroids with respect to

        :param n_jobs:
        :param kargs:
        :return:
        '''

        # Create the clustering object
        from sklearn.cluster import KMeans
        kmeans = KMeans(random_state=self.rs)
        kmeans.set_params(**self.kargs)

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

        # Loop over the other classes under picking at random
        print('Finding cluster centroids...', end="")
        for key in self.ucd.keys():
            # If the minority class is up, skip it.
            if key == self.minc:
                continue

            # Set the number of clusters to be no more than the number of samples
            if self.ratio * self.ucd[self.minc] > self.ucd[key]:
                nclusters =  self.ucd[key]
            else:
                nclusters = int(self.ratio * self.ucd[self.minc])

            # Set the number of clusters and find the centroids
            kmeans.set_params(n_clusters = nclusters)
            kmeans.fit(self.x[self.y == key])
            centroids = kmeans.cluster_centers_

            # Concatenate to the minority class
            underx = concatenate((underx, centroids), axis = 0)
            undery = concatenate((undery, ones(nclusters) * key), axis = 0)
            print(".", end="")

        print("done!")

        return underx, undery


# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
#                                   Minority over sampling children!
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
class OverSampler(UnbalancedDataset):
    '''
    Object to over-sample the minority class(es) by picking samples at random with replacement.

    *Supports multiple classes.
    '''

    def __init__(self, ratio = 1, random_state = None):
        '''
        :param ratio:
            Number of samples to draw with respect to the number of samples in the original minority class.
                N_new =

        :param random_state:
            Seed.

        :return:
            Nothing.
        '''
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

    def resample(self):
        '''
        Over samples the minority class by randomly picking samples with replacement.

        :param ratio:
            The ratio of minority elements with respect to the number of majority cases.

        :return:
            overx, overy: The features and target values of the over-sampled data set.
        '''

        # Start with the majority class
        overx = self.x[self.y == self.maxc]
        overy = self.y[self.y == self.maxc]

        # Loop over the other classes over picking at random
        for key in self.ucd.keys():
            if key == self.maxc:
                continue

            # If the ratio given is too large such that the minority becomes a majority, clip it.
            if self.ratio * self.ucd[key] > self.ucd[self.maxc]:
                num_samples = self.ucd[self.maxc] - self.ucd[key]
            else:
                num_samples = int(self.ratio * self.ucd[key])

            # Pick some elements at random
            seed(self.rs)
            indx = randint(low = 0, high = self.ucd[key], size = num_samples)

            # Concatenate to the majority class
            overx = concatenate((overx, self.x[self.y == key], self.x[self.y == key][indx]), axis=0)
            overy = concatenate((overy, self.y[self.y == key], self.y[self.y == key][indx]), axis=0)

        # Return over sampled dataset
        return overx, overy


# -------------------------------- // -------------------------------- #
class SMOTE(UnbalancedDataset):
    '''
    An implementation of SMOTE - Synthetic Minority Over-sampling Technique.

    See the original paper: SMOTE - "SMOTE: synthetic minority over-sampling technique" by Chawla, N.V et al.
    for more details.

    * Does not support multiple classes automatically, but can be called multiple times
    '''

    def __init__(self, k=5, ratio = 1, random_state = None):
        '''
        :param k:
            Number of nearest neighbours to use when constructing the synthetic samples.

        :param ratio:
            Fraction of the number of minority samples to synthetically generate.

        :param random_state:
            Seed.

        :return:
            The resampled data set with synthetic samples concatenated at the end.
        '''
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

        # Instance variable to store the number of neighbours to use.
        self.k = k

    def resample(self):
        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Finding nns
        from sklearn.neighbors import NearestNeighbors

        print("Finding the %i nearest neighbours..." % self.k, end = "")

        NN = NearestNeighbors(n_neighbors = self.k + 1)
        NN.fit(minx)
        nns = NN.kneighbors(minx, return_distance=False)[:, 1:]

        print("done!")

        # Creating synthetic samples
        print("Creating synthetic samples...", end="")
        sx, sy = make_samples(minx, minx, self.minc, nns, int(self.ratio * len(miny)), random_state=self.rs)
        print("done!")

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx), axis = 0)
        ret_y = concatenate((self.y, sy), axis = 0)

        return ret_x, ret_y


# -------------------------------- // -------------------------------- #
class bSMOTE1(UnbalancedDataset):
    '''
    An implementation of bSMOTE type 1 - Borderline Synthetic Minority Over-sampling Technique - type 1.

    See the original paper: "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning,
    by Hui Han, Wen-Yuan Wang, Bing-Huan Mao" for more details.

    * Does not support multiple classes automatically, but can be called multiple times
    '''

    def __init__(self, k=5, m=10, ratio = 1, random_state = None):
        '''
        :param k:
            The number of nearest neighbours to use to construct the synthetic samples.

        :param m:
            The number of nearest neighbours to use to determine if a minority sample is in danger.

        :param ratio:
            Fraction of the number of minority samples to synthetically generate.

        :param random_state:
            Seed.

        :return:
            The resampled data set with synthetic samples concatenated at the end.
        '''
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

        self.k = k # NN for synthetic samples
        self.m = m # NN for in_danger?

    def resample(self):
        from sklearn.neighbors import NearestNeighbors

        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Find the NNs for all samples in the data set.
        print("Finding the %i nearest neighbours..." % self.m, end = "")
        NN = NearestNeighbors(n_neighbors = self.m + 1)
        NN.fit(self.x)

        print("done!")

        # Boolean array with True for minority samples in danger
        index = asarray([in_danger(x, self.y, self.m, miny[0], NN) for x in minx])

        # If all minority samples are safe, return the original data set.
        if not any(index):
            print('There are no samples in danger. No borderline synthetic samples created.')
            return self.x, self.y

        # Find the NNs among the minority class
        NN.set_params(**{'n_neighbors' : self.k + 1})
        NN.fit(minx)
        nns = NN.kneighbors(minx[index], return_distance=False)[:, 1:]

        # Create synthetic samples for borderline points.
        sx, sy = make_samples(minx[index], minx, miny[0], nns, int(self.ratio * len(miny)), random_state=self.rs)

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx), axis = 0)
        ret_y = concatenate((self.y, sy), axis = 0)

        return ret_x, ret_y


# -------------------------------- // -------------------------------- #
class bSMOTE2(UnbalancedDataset):
    '''
    An implementation of bSMOTE type 2 - Borderline Synthetic Minority Over-sampling Technique - type 2.

    See the original paper: "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning,
    by Hui Han, Wen-Yuan Wang, Bing-Huan Mao" for more details.

    * Does not support multiple classes automatically, but can be called multiple times
    '''

    def __init__(self, k=5, m=10, ratio = 1, random_state = None):
        '''
        :param k:
            The number of nearest neighbours to use to construct the synthetic samples.

        :param m:
            The number of nearest neighbours to use to determine if a minority sample is in danger.

        :param ratio:
            Fraction of the number of minority samples to synthetically generate.

        :param random_state:
            Seed.

        :return:
            The resampled data set with synthetic samples concatenated at the end.
        '''

        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

        self.k = k # NN for synthetic samples
        self.m = m # NN for in_danger?

    def resample(self):
        from sklearn.neighbors import NearestNeighbors

        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Find the NNs for all samples in the data set.
        print("Finding the %i nearest neighbours..." % self.m, end = "")
        NN = NearestNeighbors(n_neighbors = self.m + 1)
        NN.fit(self.x)

        print("done!")

        # Boolean array with True for minority samples in danger
        index = asarray([in_danger(x, self.y, self.m, self.minc, NN) for x in minx])

        # If all minority samples are safe, return the original data set.
        if not any(index):
            print('There are no samples in danger. No borderline synthetic samples created.')
            return self.x, self.y

        # Find the NNs among the minority class
        NN.set_params(**{'n_neighbors' : self.k + 1})
        NN.fit(minx)
        nns = NN.kneighbors(minx[index], return_distance=False)[:, 1:]


        # Split the number of synthetic samples between only minority (type 1), or
        # minority and majority (with reduced step size) (type 2.
        Pyseed(self.rs)
        fractions = min(max(gauss(0.5, 0.1), 0), 1)

        # Only minority
        sx1, sy1 = make_samples(minx[index], minx, self.minc, nns,\
                                fractions * (int(self.ratio * len(miny)) + 1),\
                                step_size=1,\
                                random_state=self.rs)

        # Only majority with smaller step size
        sx2, sy2 = make_samples(minx[index], self.x[self.y != self.minc], self.minc, nns,\
                                (1 - fractions) * int(self.ratio * len(miny)),\
                                step_size=0.5,\
                                random_state=self.rs)

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx1, sx2), axis = 0)
        ret_y = concatenate((self.y, sy1, sy2), axis = 0)

        return ret_x, ret_y


# -------------------------------- // -------------------------------- #
class SVM_SMOTE(UnbalancedDataset):
    '''
    Implementation of support vector borderline SMOTE.

    Similar to borderline SMOTE it only created synthetic samples for borderline samples,
    however it looks for borderline samples by fitting and SVM classifier and
    identifying the support vectors.

    See the paper: "Borderline Over-sampling for Imbalanced Data Classification,
    by Nguyen, Cooper, Kamei"

    * Does not support multiple classes, however it can be called multiple times (I believe).
    '''

    def __init__(self, k=5, m=10, out_step=0.5, svm_args = {}, ratio = 1, random_state = None):
        '''

        :param k:
            Number of nearest neighbours to used to construct synthetic samples.

        :param m:
            The number of nearest neighbours to use to determine if a minority sample is in danger.

        :param ratio:
            Fraction of the number of minority samples to synthetically generate.

        :param out_step:
            Step size when extrapolating


        :param svm_args:
            Arguments to pass to the scikit-learn SVC object.

        :param random_state:
            Seed

        :return:
            Nothing.
        '''
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

        self.k = k
        self.m = m
        self.out_step = out_step
        self.svm_args = svm_args

    def resample(self):
        from sklearn.svm import SVC
        from sklearn.neighbors import NearestNeighbors

        svc = SVC()
        svc.set_params(**self.svm_args)

        # Fit SVM and find the support vectors
        svc.fit(self.x, self.y)
        support_index = svc.support_[self.y[svc.support_] == self.minc]
        support_vetor = self.x[support_index]

        # Start with the minority class
        minx = self.x[self.y == self.minc]

        # First, find the NN of all the samples to identify samples in danger and noisy ones
        print("Finding the %i nearest neighbours..." % self.m, end = "")
        NN = NearestNeighbors(n_neighbors = self.m + 1)
        NN.fit(self.x)
        print("done!")

        # Now, get rid of noisy support vectors

        # Boolean array with True for noisy support vectors
        noise_bool = asarray([is_noise(x, self.y, self.m, self.minc, NN) for x in support_vetor])

        # Remove noisy support vectors
        support_vetor = support_vetor[logical_not(noise_bool)]

        # Find support_vectors there are in danger (interpolation) or not (extrapolation)
        danger_bool = asarray([in_danger(x, self.y, self.m, self.minc, NN) for x in support_vetor])
        safety_bool = logical_not(danger_bool)


        print_stats = (len(support_vetor), nsum(noise_bool), nsum(danger_bool), nsum(safety_bool))
        print("Out of %i support vectors, %i are noisy, %i are in danger and %i are safe." % print_stats)

        # Proceed to find support vectors NNs among the minority class
        print("Finding the %i nearest neighbours..." % self.k, end = "")
        NN.set_params(**{'n_neighbors' : self.k + 1})
        NN.fit(minx)
        print("done!")


        print("Creating synthetic samples...", end = "")
        # Split the number of synthetic samples between interpolation and extrapolation
        Pyseed(self.rs)
        fractions = min(max(gauss(0.5, 0.1), 0), 1)

        # Interpolate samples in danger
        nns = NN.kneighbors(support_vetor[danger_bool], return_distance=False)[:, 1:]

        sx1, sy1 = make_samples(support_vetor[danger_bool], minx, self.minc, nns,\
                                fractions * (int(self.ratio * len(minx)) + 1),\
                                step_size=1,\
                                random_state=self.rs)

        # Extrapolate safe samples
        nns = NN.kneighbors(support_vetor[safety_bool], return_distance=False)[:, 1:]

        sx2, sy2 = make_samples(support_vetor[safety_bool], minx, self.minc, nns,\
                                (1 - fractions) * int(self.ratio * len(minx)),\
                                step_size=-self.out_step,\
                                random_state=self.rs)

        print("done!")

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx1, sx2), axis=0)
        ret_y = concatenate((self.y, sy1, sy2), axis=0)

        return ret_x, ret_y
    
    
    
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
#                                            Pipeline!!!
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
class Pipeline:
    '''
    A helper object to concatenate a number of re sampling objects and streamline the
    resampling process.
    '''
    
    def __init__(self, X, y):
        '''
        :param X:
            Feature matrix.

        :param y:
            Target vectors.
        '''
        
        self.x = X
        self.y = y

    def pipeline(self, list_methods):
        '''
        :param list_methods:
            Pass the methods to be used in a list, in the order they will be used.

        :return:
            The resampled dataset.
        '''

        # Initialize with the original dataset.
        x, y = self.x, self.y

        # Go through the list of methods and fit_transform each to the result of the last.
        for met in list_methods:
            x, y = met.fit_transform(x, y)
            print(x.shape)

        # Return the resampled dataset.
        return x, y