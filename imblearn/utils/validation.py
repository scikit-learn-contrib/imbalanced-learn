"""Utilities for input validation"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT
from collections import Counter

import numpy as np

from sklearn.neighbors.base import KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import six

from ..exceptions import raise_isinstance_error

SAMPLING_KIND = ('over-sampling', 'under-sampling')
RATIO_KIND = ('minority', 'majority', 'not minority', 'all', 'auto')


def check_neighbors_object(nn_name, nn_object, additional_neighbor=0):
    """Check the objects is consistent to be a NN.

    Several methods in imblearn relies on NN. Until version 0.4, these
    objects can be passed at initialisation as an integer or a
    KNeighborsMixin. After only KNeighborsMixin will be accepted. This
    utility allows for type checking and raise if the type is wrong.

    Parameters
    ----------
    nn_name : str,
        The name associated to the object to raise an error if needed.

    nn_object : int or KNeighborsMixin,
        The object to be checked

    additional_neighbor : int, optional (default=0)
        Sometimes, some algorithm need an additional neighbors.

    Returns
    -------
    nn_object : KNeighborsMixin
        The k-NN object.
    """
    if isinstance(nn_object, int):
        return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
    elif isinstance(nn_object, KNeighborsMixin):
        return nn_object
    else:
        raise_isinstance_error(nn_name, [int, KNeighborsMixin], nn_object)


def _ratio_all(y, sampling_type):
    """Returns ratio by targeting all classes."""
    target_stats = Counter(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        ratio = {key: n_sample_majority - value
                 for (key, value) in target_stats.items()}
    elif sampling_type == 'under-sampling':
        n_sample_minority = min(target_stats.values())
        ratio = {key: n_sample_minority for key in target_stats.keys()}

    return ratio


def _ratio_majority(y, sampling_type):
    """Returns ratio by targeting the majority class only."""
    if sampling_type == 'over-sampling':
        raise ValueError("'ratio'='majority' can be used with over-sampler.")
    elif sampling_type == 'under-sampling':
        target_stats = Counter(y)
        class_majority = max(target_stats, key=target_stats.get)
        n_sample_minority = min(target_stats.values())
        ratio = {key: n_sample_minority
                 for key in target_stats.keys()
                 if key == class_majority}

    return ratio


def _ratio_not_minority(y, sampling_type):
    """Returns ratio by targeting all classes but not the minority."""
    target_stats = Counter(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        ratio = {key: n_sample_majority - value
                 for (key, value) in target_stats.items()
                 if key != class_minority}
    elif sampling_type == 'under-sampling':
        n_sample_minority = min(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        ratio = {key: n_sample_minority
                 for key in target_stats.keys()
                 if key != class_minority}

    return ratio


def _ratio_minority(y, sampling_type):
    """Returns ratio by targeting the minority class only."""
    target_stats = Counter(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        ratio = {key: n_sample_majority - value
                 for (key, value) in target_stats.items()
                 if key == class_minority}
    elif sampling_type == 'under-sampling':
        raise ValueError("'ratio'='minority' can be used with under-sampler.")

    return ratio


def check_ratio(ratio, y, sampling_type):
    """Ratio validation for samplers.

    Checks ratio for consistent type and return a dictionary
    containing each targeted class with its corresponding number of
    pixel.

    Parameters
    ----------
    ratio : str, dict or callable,
        Input ratio.

        - If ``str``, has to be one of: (i) ``'minority'``: resample
          the minority class; (ii) ``'majority'``: resample the
          majority class, (i) ``'not minority'``: resample all classes
          apart of the minority class, and (i) ``'all'``: resample all
          classes.
        - If ``dict``, key is the class target and value is either the
          desired number of samples or the ratio corresponding to the
          desired number of samples over the original number of
          samples.
        - If callable, it corresponds to a function which will define
          the sampling behaviour given ``y``. It should return a
          dictionary with the key being the class target and the value
          being the desired number of samples.

    y : ndarray, shape (n_samples,)
        The target array.

    sampling_type : str,
        The type of sampling. Can be either ``'over-sampling'`` or
        ``'under-sampling'``.

    Returns
    -------
    ratio_converted : dict,
        The converted and validated ratio. Returns a dictionary with
        the key being the class target and the value being the desired
        number of samples.

    """
    if sampling_type not in SAMPLING_KIND:
        raise ValueError("'sampling_type' should be one of {}. Got '{}'"
                         " instead.".format(SAMPLING_KIND, sampling_type))

    if np.unique(y).size <= 1:
        raise ValueError("The target 'y' needs to have more than 1 class."
                         " Got {} class instead".format(np.unique(y).size))

    if isinstance(ratio, six.string_types):
        if ratio not in RATIO_KIND:
            raise ValueError("When 'ratio' is a string, it needs to be one of"
                             " {}. Got '{}' instead.".format(RATIO_KIND,
                                                             ratio))
        if ratio == 'all' or ratio == 'auto':
            ratio = _ratio_all(y, sampling_type)
        elif ratio == 'majority':
            ratio = _ratio_majority(y, sampling_type)
        elif ratio == 'not minority':
            ratio = _ratio_not_minority(y, sampling_type)
        elif ratio == 'minority':
            ratio = _ratio_minority(y, sampling_type)

    return ratio
