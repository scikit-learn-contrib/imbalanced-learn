"""Utilities for input validation"""

import warnings

from sklearn.utils.multiclass import type_of_target

def check_target_type(estimator, y):
    """Check that the estimators handle the target type provided.

    Checks which type of target is provided and if the estimator can handle
    such type.

    Parameters
    ----------
    estimator : estimator instance.
        Estimator instance for which the check is performed.

    y : ndarray, shape (n_samples, )
        Target vector which need to be checked.

    Returns
    -------
    None

    """

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not estimator._estimator_type == 'sampler':
        raise TypeError("%s is not a sampler instance." % (estimator))

    # In the case that the estimator should handle multiclass
    if estimator.get_properties()['handles_multiclass']:
        if not (type_of_target(y) == 'binary' or
                type_of_target(y) == 'multiclass'):
            warnings.warn('The target type should be binary or multiclass.')
    # In the case that the estimator is only handling binary class
    else:
        if not type_of_target(y) == 'binary':
            warnings.warn('The target type should be binary.')

    return None
