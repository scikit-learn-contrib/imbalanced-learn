"""Utilities for input validation"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT
from __future__ import division

import warnings
from collections import OrderedDict
from numbers import Integral, Real

import numpy as np

from sklearn.base import clone
from sklearn.neighbors.base import KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import six
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.deprecation import deprecated

from ..exceptions import raise_isinstance_error

SAMPLING_KIND = ('over-sampling', 'under-sampling', 'clean-sampling',
                 'ensemble', 'bypass')
TARGET_KIND = ('binary', 'multiclass', 'multilabel-indicator')


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
    if isinstance(nn_object, Integral):
        return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
    elif isinstance(nn_object, KNeighborsMixin):
        return clone(nn_object)
    else:
        raise_isinstance_error(nn_name, [int, KNeighborsMixin], nn_object)


def _count_class_sample(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def check_target_type(y, indicate_one_vs_all=False):
    """Check the target types to be conform to the current samplers.

    The current samplers should be compatible with ``'binary'``,
    ``'multilabel-indicator'`` and ``'multiclass'`` targets only.

    Parameters
    ----------
    y : ndarray,
        The array containing the target.

    indicate_one_vs_all : bool, optional
        Either to indicate if the targets are encoded in a one-vs-all fashion.

    Returns
    -------
    y : ndarray,
        The returned target.

    is_one_vs_all : bool, optional
        Indicate if the target was originally encoded in a one-vs-all fashion.
        Only returned if ``indicate_multilabel=True``.

    """
    type_y = type_of_target(y)
    if type_y not in TARGET_KIND:
        # FIXME: perfectly we should raise an error but the sklearn API does
        # not allow for it
        warnings.warn("'y' should be of types {} only. Got {} instead.".format(
            TARGET_KIND, type_of_target(y)))

    if indicate_one_vs_all:
        return (y.argmax(axis=1) if type_y == 'multilabel-indicator' else y,
                type_y == 'multilabel-indicator')
    else:
        return y.argmax(axis=1) if type_y == 'multilabel-indicator' else y


def _sampling_strategy_all(y, sampling_type):
    """Returns sampling target by targeting all classes."""
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
        }
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        n_sample_minority = min(target_stats.values())
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys()
        }
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_majority(y, sampling_type):
    """Returns sampling target by targeting the majority class only."""
    if sampling_type == 'over-sampling':
        raise ValueError("'sampling_strategy'='majority' cannot be used with"
                         " over-sampler.")
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        target_stats = _count_class_sample(y)
        class_majority = max(target_stats, key=target_stats.get)
        n_sample_minority = min(target_stats.values())
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys() if key == class_majority
        }
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_not_majority(y, sampling_type):
    """Returns sampling target by targeting all classes but not the
    majority."""
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items() if key != class_majority
        }
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        n_sample_minority = min(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys() if key != class_majority
        }
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_not_minority(y, sampling_type):
    """Returns sampling target by targeting all classes but not the
    minority."""
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items() if key != class_minority
        }
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        n_sample_minority = min(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys() if key != class_minority
        }
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_minority(y, sampling_type):
    """Returns sampling target by targeting the minority class only."""
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items() if key == class_minority
        }
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        raise ValueError("'sampling_strategy'='minority' cannot be used with"
                         " under-sampler and clean-sampler.")
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_auto(y, sampling_type):
    """Returns sampling target auto for over-sampling and not-minority for
    under-sampling."""
    if sampling_type == 'over-sampling':
        return _sampling_strategy_not_majority(y, sampling_type)
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        return _sampling_strategy_not_minority(y, sampling_type)


def _sampling_strategy_dict(sampling_strategy, y, sampling_type):
    """Returns sampling target by converting the dictionary depending of the
    sampling."""
    target_stats = _count_class_sample(y)
    # check that all keys in sampling_strategy are also in y
    set_diff_sampling_strategy_target = (
        set(sampling_strategy.keys()) - set(target_stats.keys()))
    if len(set_diff_sampling_strategy_target) > 0:
        raise ValueError("The {} target class is/are not present in the"
                         " data.".format(set_diff_sampling_strategy_target))
    # check that there is no negative number
    if any(n_samples < 0 for n_samples in sampling_strategy.values()):
        raise ValueError("The number of samples in a class cannot be negative."
                         "'sampling_strategy' contains some negative value: {}"
                         .format(sampling_strategy))
    sampling_strategy_ = {}
    if sampling_type == 'over-sampling':
        n_samples_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        for class_sample, n_samples in sampling_strategy.items():
            if n_samples < target_stats[class_sample]:
                raise ValueError("With over-sampling methods, the number"
                                 " of samples in a class should be greater"
                                 " or equal to the original number of samples."
                                 " Originally, there is {} samples and {}"
                                 " samples are asked.".format(
                                     target_stats[class_sample], n_samples))
            if n_samples > n_samples_majority:
                warnings.warn("After over-sampling, the number of samples ({})"
                              " in class {} will be larger than the number of"
                              " samples in the majority class (class #{} ->"
                              " {})".format(n_samples, class_sample,
                                            class_majority,
                                            n_samples_majority))
            sampling_strategy_[class_sample] = (
                n_samples - target_stats[class_sample])
    elif sampling_type == 'under-sampling':
        for class_sample, n_samples in sampling_strategy.items():
            if n_samples > target_stats[class_sample]:
                raise ValueError("With under-sampling methods, the number of"
                                 " samples in a class should be less or equal"
                                 " to the original number of samples."
                                 " Originally, there is {} samples and {}"
                                 " samples are asked.".format(
                                     target_stats[class_sample], n_samples))
            sampling_strategy_[class_sample] = n_samples
    elif sampling_type == 'clean-sampling':
        # FIXME: Turn into an error in 0.6
        warnings.warn("'sampling_strategy' as a dict for cleaning methods is "
                      "deprecated and will raise an error in version 0.6. "
                      "Please give a list of the classes to be targeted by the"
                      " sampling.", DeprecationWarning)
        # clean-sampling can be more permissive since those samplers do not
        # use samples
        for class_sample, n_samples in sampling_strategy.items():
            sampling_strategy_[class_sample] = n_samples
    else:
        raise NotImplementedError

    return sampling_strategy_


def _sampling_strategy_list(sampling_strategy, y, sampling_type):
    """With cleaning methods, sampling_strategy can be a list to target the
 class of interest."""
    if sampling_type != 'clean-sampling':
        raise ValueError("'sampling_strategy' cannot be a list for samplers "
                         "which are not cleaning methods.")

    target_stats = _count_class_sample(y)
    # check that all keys in sampling_strategy are also in y
    set_diff_sampling_strategy_target = (
        set(sampling_strategy) - set(target_stats.keys()))
    if len(set_diff_sampling_strategy_target) > 0:
        raise ValueError("The {} target class is/are not present in the"
                         " data.".format(set_diff_sampling_strategy_target))

    return {
        class_sample: min(target_stats.values())
        for class_sample in sampling_strategy
    }


def _sampling_strategy_float(sampling_strategy, y, sampling_type):
    """Take a proportion of the majority (over-sampling) or minority
    (under-sampling) class in binary classification."""
    type_y = type_of_target(y)
    if type_y != 'binary':
        raise ValueError(
            '"sampling_strategy" can be a float only when the type '
            'of target is binary. For multi-class, use a dict.')
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy_ = {
            key: int(n_sample_majority * sampling_strategy - value)
            for (key, value) in target_stats.items() if key != class_majority
        }
    elif (sampling_type == 'under-sampling'):
        n_sample_minority = min(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy_ = {
            key: int(n_sample_minority / sampling_strategy)
            for (key, value) in target_stats.items() if key != class_minority
        }
    else:
        raise ValueError("'clean-sampling' methods do let the user "
                         "specify the sampling ratio.")
    return sampling_strategy_


def check_sampling_strategy(sampling_strategy, y, sampling_type, **kwargs):
    """Sampling target validation for samplers.

    Checks that ``sampling_strategy`` is of consistent type and return a
    dictionary containing each targeted class with its corresponding
    number of sample. It is used in :class:`imblearn.base.BaseSampler`.

    Parameters
    ----------
    sampling_strategy : float, str, dict, list or callable,
        Sampling information to sample the data set.

        - When ``float``:

            For **under-sampling methods**, it corresponds to the ratio
            :math:`\\alpha_{us}` defined by :math:`N_{rM} = \\alpha_{us}
            \\times N_{m}` where :math:`N_{rM}` and :math:`N_{m}` are the
            number of samples in the majority class after resampling and the
            number of samples in the minority class, respectively;

            For **over-sampling methods**, it correspond to the ratio
            :math:`\\alpha_{os}` defined by :math:`N_{rm} = \\alpha_{os}
            \\times N_{m}` where :math:`N_{rm}` and :math:`N_{M}` are the
            number of samples in the minority class after resampling and the
            number of samples in the majority class, respectively.

            .. warning::
               ``float`` is only available for **binary** classification. An
               error is raised for multi-class classification and with cleaning
               samplers.

        - When ``str``, specify the class targeted by the resampling. For
          **under- and over-sampling methods**, the number of samples in the
          different classes will be equalized. For **cleaning methods**, the
          number of samples will not be equal. Possible choices are:

            ``'minority'``: resample only the minority class;

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: for under-sampling methods, equivalent to ``'not
            minority'`` and for over-sampling methods, equivalent to ``'not
            majority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

          .. warning::
             ``dict`` is available for both **under- and over-sampling
             methods**. An error is raised with **cleaning methods**. Use a
             ``list`` instead.

        - When ``list``, the list contains the targeted classes. It used only
          for **cleaning methods**.

          .. warning::
             ``list`` is available for **cleaning methods**. An error is raised
             with **under- and over-sampling methods**.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.

    y : ndarray, shape (n_samples,)
        The target array.

    sampling_type : str,
        The type of sampling. Can be either ``'over-sampling'``,
        ``'under-sampling'``, or ``'clean-sampling'``.

    kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to
        ``sampling_strategy`` when this is a callable.

    Returns
    -------
    sampling_strategy_converted : dict,
        The converted and validated sampling target. Returns a dictionary with
        the key being the class target and the value being the desired
        number of samples.

    """
    if sampling_type not in SAMPLING_KIND:
        raise ValueError("'sampling_type' should be one of {}. Got '{}'"
                         " instead.".format(SAMPLING_KIND, sampling_type))

    if np.unique(y).size <= 1:
        raise ValueError("The target 'y' needs to have more than 1 class."
                         " Got {} class instead".format(np.unique(y).size))

    if sampling_type in ('ensemble', 'bypass'):
        return sampling_strategy

    if isinstance(sampling_strategy, six.string_types):
        if sampling_strategy not in SAMPLING_TARGET_KIND.keys():
            raise ValueError("When 'sampling_strategy' is a string, it needs"
                             " to be one of {}. Got '{}' instead.".format(
                                 SAMPLING_TARGET_KIND, sampling_strategy))
        return OrderedDict(sorted(
            SAMPLING_TARGET_KIND[sampling_strategy](y, sampling_type).items()))
    elif isinstance(sampling_strategy, dict):
        return OrderedDict(sorted(
            _sampling_strategy_dict(sampling_strategy, y, sampling_type)
            .items()))
    elif isinstance(sampling_strategy, list):
        return OrderedDict(sorted(
            _sampling_strategy_list(sampling_strategy, y, sampling_type)
            .items()))
    elif isinstance(sampling_strategy, Real):
        if sampling_strategy <= 0 or sampling_strategy > 1:
            raise ValueError(
                "When 'sampling_strategy' is a float, it should be "
                "in the range (0, 1]. Got {} instead."
                .format(sampling_strategy))
        return OrderedDict(sorted(
            _sampling_strategy_float(sampling_strategy, y, sampling_type)
            .items()))
    elif callable(sampling_strategy):
        sampling_strategy_ = sampling_strategy(y, **kwargs)
        return OrderedDict(sorted(
            _sampling_strategy_dict(sampling_strategy_, y, sampling_type)
            .items()))


SAMPLING_TARGET_KIND = {
    'minority': _sampling_strategy_minority,
    'majority': _sampling_strategy_majority,
    'not minority': _sampling_strategy_not_minority,
    'not majority': _sampling_strategy_not_majority,
    'all': _sampling_strategy_all,
    'auto': _sampling_strategy_auto
}


@deprecated("imblearn.utils.check_ratio was deprecated in favor of "
            "imblearn.utils.check_sampling_strategy in 0.4. It will be "
            "removed in 0.6.")
def check_ratio(ratio, y, sampling_type, **kwargs):
    """Sampling target validation for samplers.

    Checks ratio for consistent type and return a dictionary
    containing each targeted class with its corresponding number of
    sample.

    .. deprecated:: 0.4
       This function is deprecated in favor of
       :func:`imblearn.utils.check_sampling_strategy`. It will be removed in
       0.6.

    Parameters
    ----------
    ratio : str, dict or callable,
        Ratio to use for resampling the data set.

        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    y : ndarray, shape (n_samples,)
        The target array.

    sampling_type : str,
        The type of sampling. Can be either ``'over-sampling'`` or
        ``'under-sampling'``.

    kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to ``ratio``.

    Returns
    -------
    ratio_converted : dict,
        The converted and validated ratio. Returns a dictionary with
        the key being the class target and the value being the desired
        number of samples.

    """
    return check_sampling_strategy(ratio, y, sampling_type, **kwargs)
