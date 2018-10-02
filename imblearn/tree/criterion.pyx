# Author: Evgeni Dubov <evgeni.dubov@gmail.com>
#
# License: BSD 3 clause

from libc.math cimport sqrt, pow
from libc.math cimport abs

import numpy as np

from sklearn.tree._criterion cimport ClassificationCriterion
from sklearn.tree._criterion cimport SIZE_t

cdef double INFINITY = np.inf


cdef class HellingerDistanceCriterion(ClassificationCriterion):
    """Hellinger distance criterion.


    """

    cdef double proxy_impurity_improvement(self) nogil:
        cdef:
            double impurity_left
            double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return impurity_right + impurity_left

    cdef double impurity_improvement(self, double impurity) nogil:
        cdef:
            double impurity_left
            double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return impurity_right + impurity_left

    cdef double node_impurity(self) nogil:
        cdef:
            SIZE_t* n_classes = self.n_classes
            double* sum_total = self.sum_total
            double hellinger = 0.0
            double sq_count
            double count_k
            SIZE_t k, c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                hellinger += 1.0

        return hellinger / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef:
            SIZE_t* n_classes = self.n_classes
            double* sum_left = self.sum_left
            double* sum_right = self.sum_right
            double hellinger_left = 0.0
            double hellinger_right = 0.0
            double count_k1 = 0.0
            double count_k2 = 0.0
            SIZE_t k, c

        # stop splitting in case reached pure node with 0 samples of second
        # class
        if sum_left[1] + sum_right[1] == 0:
            impurity_left[0] = -INFINITY
            impurity_right[0] = -INFINITY
            return

        for k in range(self.n_outputs):
            if(sum_left[0] + sum_right[0] > 0):
                count_k1 = sqrt(sum_left[0] / (sum_left[0] + sum_right[0]))
            if(sum_left[1] + sum_right[1] > 0):
                count_k2 = sqrt(sum_left[1] / (sum_left[1] + sum_right[1]))

            hellinger_left += pow((count_k1  - count_k2), 2)

            if(sum_left[0] + sum_right[0] > 0):
                count_k1 = sqrt(sum_right[0] / (sum_left[0] + sum_right[0]))
            if(sum_left[1] + sum_right[1] > 0):
                count_k2 = sqrt(sum_right[1] / (sum_left[1] + sum_right[1]))

            hellinger_right += pow((count_k1  - count_k2), 2)

        impurity_left[0]  = hellinger_left  / self.n_outputs
        impurity_right[0] = hellinger_right / self.n_outputs
