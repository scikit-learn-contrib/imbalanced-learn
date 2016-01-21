#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: mithril
# @Date:   2016-01-20 17:12:13
# @Last Modified by:   mithril
# @Last Modified time: 2016-01-21 10:42:47

__auther__ = "eromoe@mithril"

import numpy as np
import scipy.sparse as sp

def concatenate(l, axis=0):
    """
    :param l:
        tuple or list of array

    :param axis:

    :return:
        array
    """
    if not isinstance(l, (tuple, list)):
        raise Exception('concatenate need tuple like input')

    if all(isinstance(x, np.ndarray) for x in l):
        return np.concatenate(l, axis)
    else:
        for x in l:
            if not isinstance(x, sp.csr.csr_matrix):
                x = sp.csr_matrix(x)

        if axis:
            return sp.hstack(l)
        else:
            return sp.vstack(l)


    # raise Exception('You input is unable to concatenate. The input objects must all be numpy.ndarray or scipy.sparse.csr.csr_matrix, check detail:%s' % e)