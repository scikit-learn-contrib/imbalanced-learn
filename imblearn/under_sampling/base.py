"""
Base class for the under-sampling method.
"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from ..base import BaseSampler


class BaseUnderSampler(BaseSampler):
    """Base class for under-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _sampling_type = "under-sampling"

    _sampling_strategy_docstring = """sampling_strategy : float, str, dict, callable, default='auto'
        Sampling information to sample the data set.

        - When ``float``, it corresponds to the desired ratio of the number of
          samples in the minority class over the number of samples in the
          majority class after resampling. Therefore, the ratio is expressed as
          :math:`\\alpha_{us} = N_{m} / N_{rM}` where :math:`N_{m}` is the
          number of samples in the minority class and
          :math:`N_{rM}` is the number of samples in the majority class
          after resampling.

          .. warning::
             ``float`` is only available for **binary** classification. An
             error is raised for multi-class classification.

        - When ``str``, specify the class targeted by the resampling. The
          number of samples in the different classes will be equalized.
          Possible choices are:

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not minority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
        """.rstrip()


class BaseCleaningSampler(BaseSampler):
    """Base class for under-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _sampling_type = "clean-sampling"

    _sampling_strategy_docstring = """sampling_strategy : str, list or callable
        Sampling information to sample the data set.

        - When ``str``, specify the class targeted by the resampling. Note the
          the number of samples will not be equal in each. Possible choices
          are:

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not minority'``.

        - When ``list``, the list contains the classes targeted by the
          resampling.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
        """.rstrip()
