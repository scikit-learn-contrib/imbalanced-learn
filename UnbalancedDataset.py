'''
Hello!

A class to help dealing with unbalanced data sets!

VERY UNDER CONSTRUCTION

Under-sampling
Over-sampling
SMOTE
borderline SMOTE (1 & 2)
SVM borderline SMOTE

UCC: Undersampling with Cluster Centroids (experimental)


references:

smote
"SMOTE: synthetic minority over-sampling technique" by
Chawla, N.V et al.

bsmote
Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning, Hui Han, Wen-Yuan Wang, Bing-Huan Mao

svm_smote
Borderline Over-sampling for Imbalanced Data Classification, Nguyen, Cooper, Kamei


Interesting things to do/add:
    ...
'''



from __future__ import division
from __future__ import print_function

__author__ = 'fnogueira'


#import numpy

from random import gauss
from random import seed as Pyseed

from numpy.random import seed, randint, uniform
from numpy import zeros, ones, concatenate, logical_not, asarray

# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
#                                            Functions!!!
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
def is_tomek(y, nns, class_type):

    links = zeros(len(y), dtype=bool)
    count = 0

    for ind, ele in enumerate(y):

        if ele == class_type:
            continue

        if y[nns[ind]] == class_type:

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

        new[i] = x[row] - step * (x[row] - X[nns[row, col]]) # Uses big X for neighbours!

    y_new = ones(len(new)) * ytype

    return new, y_new

# -------------------------------- // -------------------------------- #
def in_danger(sample, Y, m, class_type, nn_obj):
    '''
    Function to determine whether a given minority samples is in Danger as defined by
    Chawla, N.V et al., in: SMOTE: synthetic minority over-sampling technique.

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

    x = nn_obj.kneighbors(sample.reshape((1, len(sample))), return_distance=False)[:, 1:]

    minority = 0
    for nn in x[0]:
        if Y[nn] != class_type:
            continue
        else:
            minority += 1

    if minority <= m/2 or minority == m:
        # for minority == k the sample is considered to be noise and won't be used, similarly to safe samples
        return False
    else:
        return True


# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
#                                           Parent Class!
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
class UnbalancedDataset:

    def __init__(self, ratio=1, random_state=None):
        self.rs = random_state

        self.ucd = {}
        self.ratio = ratio

        self.x = None
        self.y = None
        self.minc = None

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

        self.out_x, self.out_y = self.resample()

        return self.out_x, self.out_y

    def fit_transform(self, X, y):

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

    def __init__(self, ratio = 1, random_state = None):
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

    def resample(self):
        '''
        Under samples the majority class (or classes) by randomly picking a fraction of the
        samples with replacement.

        :param ratio:
            The ratio of majority elements to sample with respect to the number of minority cases.

        :return:
            underx, undery: The features and target values of the under-sampled data set.
        '''

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

    def __init__(self):
        UnbalancedDataset.__init__(self)

    def resample(self):
        '''

        :return:
            Return the data with majority samples that form a Tomek link removed.
        '''

        from sklearn.neighbors import NearestNeighbors

        print("Finding nearest neighbour...", end="")
        NN = NearestNeighbors(n_neighbors = 2)
        NN.fit(self.x)
        nns = NN.kneighbors(self.x, return_distance=False)[:, 1]

        print("done!")


        print("Looking for majority Tomek links...", end="")

        links = is_tomek(self.y, nns, self.minc)


        return self.x[logical_not(links)], self.y[logical_not(links)]




# -------------------------------- // -------------------------------- #
# Experimental techniques
class ClusterCentroids(UnbalancedDataset):

    def __init__(self, kargs = {}, ratio = 1, random_state = None):
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

        self.kargs = kargs

    def resample(self):
        '''
        Experimental methods that under samples the majority class by replacing a cluster of
        majority samples by the cluster centroid of a KMeans algorithm.

        This algorithm keeps N majority samples by fitting the KMeans algorithm with N cluster
        to the majority class and using the coordinates of the N cluster centroids as the new
        majority samples.

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

    def __init__(self, ratio = 1, random_state = None):
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

    def resample(self):
        '''
        Over samples the minority class by randomly picking samples with replacement.

        :param ratio:
            The ratio of minority elements with respect to the number of majority cases.

        :return:
            overx, overy: The features and target values of the over-sampled data set.
        '''

        # Start with the minority class
        overx = self.x[self.y == self.minc]
        overy = self.y[self.y == self.minc]

        # Loop over the other classes under picking at random
        for key in self.ucd.keys():
            if key == self.maxc:
                continue


            if self.ratio * self.ucd[self.maxc] < self.ucd[key]:
                num_samples = 0
            else:
                num_samples = int(self.ratio * self.ucd[self.maxc]) - self.ucd[key]

            # Pick some elements at random
            seed(self.rs)
            indx = randint(low = 0, high = self.ucd[key], size = num_samples)

            # Concatenate to the minority class
            overx = concatenate((overx, self.x[self.y == key], self.x[self.y == key][indx]), axis = 0)
            overy = concatenate((overy, self.y[self.y == key], self.y[self.y == key][indx]), axis = 0)

        return overx, overy



# -------------------------------- // -------------------------------- #
class SMOTE(UnbalancedDataset):

    def __init__(self, k=5, ratio = 1, random_state = None):
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

        self.k = k

    def resample(self):
        '''
        Synthetic Minority Over-sampling Technique implementation.

        :param k:
            Number of nearest neighbours to construct the synthetic samples.

        :param ratio:
            Fraction of the number of minority samples to synthetically generate.

        :return:
            --- Have yet to decide on the return ---
        '''

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

        print("Creating synthetic samples...", end = "")
        # Creating synthetic samples
        sx, sy = make_samples(minx, minx, self.minc, nns, int(self.ratio * len(miny)), random_state=self.rs)
        print("done!")

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx), axis = 0)
        ret_y = concatenate((self.y, sy), axis = 0)

        return ret_x, ret_y


# -------------------------------- // -------------------------------- #
class bSMOTE1(UnbalancedDataset):

    def __init__(self, k=5, m=10, ratio = 1, random_state = None):
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

        self.k = k
        self.m = m

    def resample(self):
        '''
        Implementation of borderline synthetic minority oversampling technique.

        :param k:
            Number of nearest neighbours used to construct the synthetic samples.

        :param m:
            Number of nearest neighbours used to decide which minority samples are in danger.

        :param ratio:
            Fraction of the number of minority samples to synthetically generate.

        :return:
        --- Have yet to decide on the return ---
        '''

        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Finding nns
        from sklearn.neighbors import NearestNeighbors

        print("Finding the %i nearest neighbours..." % self.m, end = "")

        NN = NearestNeighbors(n_neighbors = self.m + 1)
        NN.fit(self.x)

        print("done!")

        index = asarray([in_danger(x, self.y, self.m, miny[0], NN) for x in minx])

        if not any(index):
            print('There are no samples in danger, no borderline synthetic samples created.')
            return self.x, self.y


        NN.set_params(**{'n_neighbors' : self.k + 1})
        NN.fit(minx)
        nns = NN.kneighbors(minx[index], return_distance=False)[:, 1:]

        sx, sy = make_samples(minx[index], minx, miny[0], nns, int(self.ratio * len(miny)), random_state=self.rs)

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx), axis = 0)
        ret_y = concatenate((self.y, sy), axis = 0)

        return ret_x, ret_y


# -------------------------------- // -------------------------------- #
class bSMOTE2(UnbalancedDataset):

    def __init__(self, k=5, m=10, ratio = 1, random_state = None):
        '''
        Implementation of borderline synthetic minority oversampling technique.

        :param k:
            Number of nearest neighbours used to construct the synthetic samples.

        :param m:
            Number of nearest neighbours used to decide which minority samples are in danger.

        :param ratio:
            Fraction of the number of minority samples to synthetically generate.

        :return:
        --- Have yet to decide on the return ---
        '''

        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

        self.k = k
        self.m = m

    def resample(self):
        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Finding nns
        from sklearn.neighbors import NearestNeighbors

        print("Finding the %i nearest neighbours..." % self.m, end = "")

        NN = NearestNeighbors(n_neighbors = self.m + 1)
        NN.fit(self.x)

        print("done!")

        index = asarray([in_danger(x, self.y, self.m, miny[0], NN) for x in minx])

        if not any(index):
            print('no samples in danger')
            return self.x, self.y


        NN.set_params(**{'n_neighbors' : self.k + 1})
        NN.fit(minx)
        nns = NN.kneighbors(minx[index], return_distance=False)[:, 1:]


        Pyseed(self.rs)
        fractions = min(max(gauss(0.5, 0.1), 0), 1)

        sx1, sy1 = make_samples(minx[index], minx, miny[0], nns,\
                                fractions * (int(self.ratio * len(miny)) + 1),\
                                random_state=self.rs)

        sx2, sy2 = make_samples(minx[index], self.x[self.y != self.minc], self.minc, nns,\
                                (1 - fractions) * int(self.ratio * len(miny)),\
                                step_size=0.5,\
                                random_state=self.rs)


        sx = concatenate((sx1, sx2), axis=0)
        sy = concatenate((sy1, sy2), axis=0)

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx), axis = 0)
        ret_y = concatenate((self.y, sy), axis = 0)

        return ret_x, ret_y


# -------------------------------- // -------------------------------- #
class SVM_SMOTE(UnbalancedDataset):

    def __init__(self, k=5, svm_args = {}, ratio = 1, random_state = None):
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

        self.k = k
        self.svm_args = svm_args

    def resample(self):
        '''
        Implementation of borderline synthetic minority oversampling technique using SVM support vectors.

        :param k:
            Number of nearest neighbours to used to construct synthetic samples.

        :param ratio:
            Fraction of the number of minority samples to synthetically generate.

        :param svm_args:
            Arguments to pass to the scikit-learn SVC object.

        :return:
            --- Have yet to decide on the return ---
        '''

        from sklearn.svm import SVC
        svc = SVC()
        svc.set_params(**self.svm_args)

        # Fit SVM and find the support vectors
        svc.fit(self.x, self.y)
        sup_min = svc.support_[self.y[svc.support_] == self.minc]

        # Start with the minority class
        minx = self.x[self.y == self.minc]
        #miny = self.y[self.y == self.minc]

        # Finding nns
        from sklearn.neighbors import NearestNeighbors

        print("Finding the %i nearest neighbours..." % self.k, end = "")

        NN = NearestNeighbors(n_neighbors = self.k + 1)
        NN.fit(minx)
        nns = NN.kneighbors(self.x[sup_min], return_distance=False)[:, 1:]

        print("done!")

        print("Creating synthetic samples...", end = "")
        # Creating synthetic samples
        sx, sy = make_samples(self.x[sup_min], minx, self.y[sup_min][0], nns, int(self.ratio * len(minx)), random_state=self.rs)
        print("done!")

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx), axis = 0)
        ret_y = concatenate((self.y, sy), axis = 0)

        return ret_x, ret_y
    
    
    
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
#                                            Pipeline!!!
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
# ---------------------------------------------- // ---------------------------------------------- #
class Pipeline:
    
    def __init__(self, X, y):
        
        self.x = X
        self.y = y

    def pipeline(self, list_methods):

        x, y = self.x, self.y

        for met in list_methods:
            x, y = met.fit_transform(x, y)
            print(x.shape)

        return x, y
        



if __name__ == '__main__':

    from sklearn.datasets import make_classification
    x, y = make_classification(n_classes=2, class_sep=1, weights=[0.9, 0.1], \
                               n_informative=2, n_redundant=0, flip_y=0,\
                               n_features=2, n_clusters_per_class=1,\
                               n_samples=500, random_state=10)


    test = UnderSampler()

    test.fit(x, y)

    a, b = test.transform()

    test2 = TomekLinks()
    c, d = test2.fit_transform(x, y)

    test3 = ClusterCentroids()
    e, f = test3.fit_transform(x, y)


    pip = Pipeline(x, y)
    g, h = pip.pipeline([TomekLinks(), ClusterCentroids()])

    i, j = pip.pipeline([TomekLinks(), SVM_SMOTE(k=6, svm_args={'class_weight' : 'auto'}), UnderSampler(random_state=1)])
    i, j = pip.pipeline([TomekLinks(), SMOTE(), UnderSampler(random_state=1)])
    i, j = pip.pipeline([TomekLinks(), bSMOTE1(), UnderSampler(random_state=1)])
    i, j = pip.pipeline([TomekLinks(), bSMOTE2(random_state=1), UnderSampler(random_state=1)])