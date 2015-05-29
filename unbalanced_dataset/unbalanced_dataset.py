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

[4] NearMiss - "kNN approach to unbalanced data distributions: A case study
involving information extraction" by Zhang et al.

[5] CNN - "Addressing the Curse of Imbalanced Training Sets: One-Sided
Selection" by Kubat et al.

[6] One-Sided Selection - "Addressing the Curse of Imbalanced Training Sets:
One-Sided Selection" by Kubat et al.

[7] NCL - "Improving identification of difficult small classes by balancing
class distribution" by Laurikkala et al.

[8] SMOTE + Tomek - "Balancing training data for automated annotation of
keywords: a case study" by Batista et al.

[9] SMOTE + ENN - "A study of the behavior of several methods for balancing
machine learning training data" by Batista et al.

[10] EasyEnsemble & BalanceCascade - "Exploratory Understanding for
Class-Imbalance Learning" by Liu et al.

TO DO LIST:
===========
"""

from __future__ import division
from __future__ import print_function
from numpy.random import seed, randint, uniform
from numpy import zeros, ones

__author__ = 'fnogueira, glemaitre'


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

        if self.verbose:
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

        if self.verbose:
            print(str(len(uniques)) +
                  " classes detected: " +
                  str(self.ucd), end="\n")

    def transform(self):
        """
        Class method to re-sample the dataset with a particular technique.

        :return:
            The re-sampled data set.
        """

        if self.verbose:
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

    @staticmethod
    def is_tomek(y, nn_index, class_type, verbose=True):
        """
        is_tomek uses the target vector and the first neighbour of every sample
        point and looks for Tomek pairs. Returning a boolean vector with True
        for majority Tomek links.

        :param y:
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority or not

        :param nn_index:
            The index of the closes nearest neighbour to a sample point.

        :param class_type:
            The label of the minority class.

        :return:
            Boolean vector on len( # samples ), with True for majority samples
            that are Tomek links.
        """

        # Initialize the boolean result as false, and also a counter
        links = zeros(len(y), dtype=bool)
        count = 0

        # Loop through each sample and looks whether it belongs to the minority
        # class. If it does, we don't consider it since we want to keep all
        # minority samples. If, however, it belongs to the majority sample we
        # look at its first neighbour. If its closest neighbour also has the
        # current sample as its closest neighbour, the two form a Tomek link.
        for ind, ele in enumerate(y):

            if ele == class_type:
                continue

            if y[nn_index[ind]] == class_type:

                # If they form a tomek link, put a True marker on this
                # sample, and increase counter by one.
                if nn_index[nn_index[ind]] == ind:
                    links[ind] = True
                    count += 1

        if verbose:
            print("%i Tomek links found." % count)

        return links

    @staticmethod
    def make_samples(x, nn_data, y_type, nn_num, n_samples,
                     step_size=1., random_state=None, verbose=True):
        """
        A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        :param x:
            Minority points for which new samples are going to be created.

        :param nn_data:
            Data set carrying all the neighbours to be used

        :param y_type:
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format

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
        seeds = randint(low=0,
                        high=100*len(nn_num.flatten()),
                        size=n_samples)

        # Randomly pick samples to construct neighbours from
        seed(random_state)
        samples = randint(low=0,
                          high=len(nn_num.flatten()),
                          size=n_samples)

        # Loop over the NN matrix and create new samples
        for i, n in enumerate(samples):
            # NN lines relate to original sample, columns to its
            # nearest neighbours
            row, col = divmod(n, len(nn_num.T))

            # Take a step of random size (0,1) in the direction of the
            # n nearest neighbours
            seed(seeds[i])
            step = step_size * uniform()

            # Construct synthetic sample
            new[i] = x[row] - step * (x[row] - nn_data[nn_num[row, col]])

        # The returned target vector is simply a repetition of the
        # minority label
        y_new = ones(len(new)) * y_type

        if verbose:
            print("Generated %i new samples ..." % len(new))

        return new, y_new

    @staticmethod
    def in_danger(entry, y, m, class_type, nn_obj):
        """
        Function to determine whether a given minority samples is in Danger as
        defined by Chawla, N.V et al., in: SMOTE: synthetic minority
        over-sampling technique.

        A minority sample is in danger if more than half of its nearest
        neighbours belong to the majority class. The exception being a
        minority sample for which all its nearest neighbours are from the
        majority class, in which case it is considered noise.

        :param entry:
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
        x = nn_obj.kneighbors(entry.reshape((1, len(entry))),
                              return_distance=False)[:, 1:]

        # Count how many NN belong to the minority class
        minority = 0
        for nn in x[0]:
            if y[nn] != class_type:
                continue
            else:
                minority += 1

        # Return True of False for in danger and not in danger or
        # noise samples.
        if minority <= m/2 or minority == m:
            # for minority == k the sample is considered to be noise and
            # won't be used, similarly to safe samples
            return False
        else:
            return True

    @staticmethod
    def is_noise(entry, y, class_type, nn_obj):
        """
        Function to determine whether a given minority sample is noise as
        defined in [1].

        A minority sample is noise if all its nearest neighbours belong to
        the majority class.

        :param entry:
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
        x = nn_obj.kneighbors(entry.reshape((1, len(entry))),
                              return_distance=False)[:, 1:]

        # Check if any neighbour belong to the minority class.
        for nn in x[0]:
            if y[nn] != class_type:
                continue
            else:
                return False

        # If the loop completed, it is noise.
        return True