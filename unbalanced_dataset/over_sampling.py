from __future__ import print_function
from __future__ import division
import numpy as np
from numpy.random import seed, randint
from numpy import concatenate, asarray
from random import betavariate
from collections import Counter
from .unbalanced_dataset import UnbalancedDataset


class OverSampler(UnbalancedDataset):
    """
    Object to over-sample the minority class(es) by picking samples at random
    with replacement.

    *Supports multiple classes.
    """

    def __init__(self, ratio=1., method='replacement', random_state=None, verbose=True, **kwargs):
        """
        :param ratio:
            Fraction of samples to draw with respect to the number of samples in
            the original minority class, e.g., if ratio=0.5 the new total size of
            minority class would be 1.5 times the original.
                N_new =

        :param random_state:
            Seed.

        :return:
            Nothing.
        """
        UnbalancedDataset.__init__(self,
                                   ratio=ratio,
                                   random_state=random_state,
                                   verbose=verbose)

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


class SMOTE(UnbalancedDataset):
    """
    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique, and the variations Borderline SMOTE 1, 2 and
    SVM-SMOTE.

    See the original papers: [1], [2], [3] for more details.

    * It does not support multiple classes automatically, but can be called
    multiple times
    """

    def __init__(self,
                 k=5,
                 m=10,
                 out_step=0.5,
                 ratio=1,
                 random_state=None,
                 kind='regular',
                 verbose=False,
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
                                       random_state=self.rs,
                                       verbose=self.verbose)

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
                sx, sy = self.make_samples(minx[danger_index],
                                           minx,
                                           miny[0],
                                           nns,
                                           int(self.ratio * len(miny)),
                                           random_state=self.rs,
                                           verbose=self.verbose)

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
                sx1, sy1 = self.make_samples(minx[danger_index],
                                             minx,
                                             self.minc,
                                             nns,
                                             fractions * (int(self.ratio * len(miny)) + 1),
                                             step_size=1,
                                             random_state=self.rs,
                                             verbose=self.verbose)

                # Only majority with smaller step size
                sx2, sy2 = self.make_samples(minx[danger_index],
                                             self.x[self.y != self.minc],
                                             self.minc, nns,
                                             (1 - fractions) * int(self.ratio * len(miny)),
                                             step_size=0.5,
                                             random_state=self.rs,
                                             verbose=self.verbose)

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
            danger_bool = [self.in_danger(x,
                                          self.y,
                                          self.m,
                                          self.minc,
                                          self.nearest_neighbour_)
                           for x in support_vector]

            # Turn into array#
            danger_bool = asarray(danger_bool)

            # Something ...#
            safety_bool = np.logical_not(danger_bool)

            if self.verbose:
                print("Out of {0} support vectors, {1} are noisy, "
                      "{2} are in danger "
                      "and {3} are safe.".format(support_vector.shape[0],
                                                 noise_bool.sum().astype(int),
                                                 danger_bool.sum().astype(int),
                                                 safety_bool.sum().astype(int)
                                                 )
                      )

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
            if (np.count_nonzero(danger_bool) > 0):
                nns = self.nearest_neighbour_.kneighbors(support_vector[danger_bool],
                                                         return_distance=False)[:, 1:]

                sx1, sy1 = self.make_samples(support_vector[danger_bool],
                                             minx,
                                             self.minc, nns,
                                             fractions * (int(self.ratio * len(minx)) + 1),
                                             step_size=1,
                                             random_state=self.rs,
                                             verbose=self.verbose)

            # Extrapolate safe samples
            if (np.count_nonzero(safety_bool) > 0):
                nns = self.nearest_neighbour_.kneighbors(support_vector[safety_bool],
                                                         return_distance=False)[:, 1:]
                
                sx2, sy2 = self.make_samples(support_vector[safety_bool],
                                             minx,
                                             self.minc, nns,
                                             (1 - fractions) * int(self.ratio * len(minx)),
                                             step_size=-self.out_step,
                                             random_state=self.rs,
                                             verbose=self.verbose)

            if self.verbose:
                print("done!")

            # Concatenate the newly generated samples to the original data set
            if (  (np.count_nonzero(danger_bool) > 0) and
                  (np.count_nonzero(safety_bool) > 0)     ):
                ret_x = concatenate((self.x, sx1, sx2), axis=0)
                ret_y = concatenate((self.y, sy1, sy2), axis=0)
            # not any support vectors in danger
            elif np.count_nonzero(danger_bool) == 0:
                ret_x = concatenate((self.x, sx2), axis=0)
                ret_y = concatenate((self.y, sy2), axis=0)
            # All the support vector in danger
            elif np.count_nonzero(safety_bool) == 0:
                ret_x = concatenate((self.x, sx1), axis=0)
                ret_y = concatenate((self.y, sy1), axis=0)

            return ret_x, ret_y
