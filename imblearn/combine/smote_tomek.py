"""Class to perform over-sampling using SMOTE and cleaning using Tomek
links."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import logging
import warnings

from sklearn.utils import check_X_y

from ..base import SamplerMixin
from ..over_sampling import SMOTE
from ..under_sampling import TomekLinks
from ..utils import check_target_type, hash_X_y


class SMOTETomek(SamplerMixin):
    """Class to perform over-sampling using SMOTE and cleaning using
    Tomek links.

    Combine over- and under-sampling using SMOTE and Tomek links.

    Read more in the :ref:`User Guide <combine>`.

    Parameters
    ----------
    ratio : str, dict, or callable, optional (default='auto')
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

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    smote : object, optional (default=SMOTE())
        The :class:`imblearn.over_sampling.SMOTE` object to use. If not given,
        a :class:`imblearn.over_sampling.SMOTE` object with default parameters
        will be given.

    tomek : object, optional (default=Tomek())
        The :class:`imblearn.under_sampling.Tomek` object to use. If not given,
        a :class:`imblearn.under_sampling.Tomek` object with default parameters
        will be given.

    k : int, optional (default=None)
        Number of nearest neighbours to used to construct synthetic
        samples.

        .. deprecated:: 0.2
           ``k`` is deprecated from 0.2 and will be replaced in 0.4
           Give directly a :class:`imblearn.over_sampling.SMOTE` object.

    m : int, optional (default=None)
        Number of nearest neighbours to use to determine if a minority
        sample is in danger.

        .. deprecated:: 0.2
           ``m`` is deprecated from 0.2 and will be replaced in 0.4
           Give directly a :class:`imblearn.over_sampling.SMOTE` object.

    out_step : float, optional (default=None)
        Step size when extrapolating.

        .. deprecated:: 0.2
           ``out_step`` is deprecated from 0.2 and will be replaced in 0.4
           Give directly a :class:`imblearn.over_sampling.SMOTE` object.

    kind_smote : str, optional (default=None)
        The type of SMOTE algorithm to use one of the following
        options: ``'regular'``, ``'borderline1'``, ``'borderline2'``,
        ``'svm'``.

        .. deprecated:: 0.2
           ``kind_smote`` is deprecated from 0.2 and will be replaced in 0.4
           Give directly a :class:`imblearn.over_sampling.SMOTE` object.

    n_jobs : int, optional (default=None)
        The number of threads to open if possible.

        .. deprecated:: 0.2
           ``n_jobs`` is deprecated from 0.2 and will be replaced in 0.4
           Give directly a :class:`imblearn.over_sampling.SMOTE` object.

    Notes
    -----
    The methos is presented in [1]_.

    Supports mutli-class resampling. Refer to SMOTE and TomekLinks regarding
    the scheme which used.

    See :ref:`sphx_glr_auto_examples_combine_plot_smote_tomek.py` and
    :ref:`sphx_glr_auto_examples_combine_plot_comparison_combine.py`.

    See also
    --------
    SMOTEENN : Over-sample using SMOTE followed by under-sampling using Edited
        Nearest Neighbours.

    References
    ----------
    .. [1] G. Batista, B. Bazzan, M. Monard, "Balancing Training Data for
       Automated Annotation of Keywords: a Case Study," In WOB, 10-18, 2003.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.combine import \
SMOTETomek # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> smt = SMOTETomek(random_state=42)
    >>> X_res, y_res = smt.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 900, 1: 900})

    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 smote=None,
                 tomek=None,
                 k=None,
                 m=None,
                 out_step=None,
                 kind_smote=None,
                 n_jobs=None):
        super(SMOTETomek, self).__init__()
        self.ratio = ratio
        self.random_state = random_state
        self.smote = smote
        self.tomek = tomek
        self.k = k
        self.m = m
        self.out_step = out_step
        self.kind_smote = kind_smote
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)

    def _validate_estimator(self):
        "Private function to validate SMOTE and ENN objects"

        # Check any parameters for SMOTE was provided
        # Anounce deprecation
        if (self.k is not None or self.m is not None or
                self.out_step is not None or self.kind_smote is not None or
                self.n_jobs is not None):
            warnings.warn('Parameters initialization will be replaced in'
                          ' version 0.4. Use a SMOTE object instead.',
                          DeprecationWarning)
            # We need to list each parameter and decide if we affect a default
            # value or not
            if self.k is None:
                self.k = 5
            if self.m is None:
                self.m = 10
            if self.out_step is None:
                self.out_step = 0.5
            if self.kind_smote is None:
                self.kind_smote = 'regular'
            if self.n_jobs is None:
                smote_jobs = 1
            else:
                smote_jobs = self.n_jobs
            self.smote_ = SMOTE(
                ratio=self.ratio,
                random_state=self.random_state,
                k=self.k,
                m=self.m,
                out_step=self.out_step,
                kind=self.kind_smote,
                n_jobs=smote_jobs)
        # If an object was given, affect
        elif self.smote is not None:
            if isinstance(self.smote, SMOTE):
                self.smote_ = self.smote
            else:
                raise ValueError('smote needs to be a SMOTE object.'
                                 'Got {} instead.'.format(type(self.smote)))
        # Otherwise create a default SMOTE
        else:
            self.smote_ = SMOTE(
                ratio=self.ratio, random_state=self.random_state)

        # Check any parameters for ENN was provided
        # Anounce deprecation
        if self.n_jobs is not None:
            warnings.warn('Parameters initialization will be replaced in'
                          ' version 0.4. Use a ENN object instead.',
                          DeprecationWarning)
            self.tomek_ = TomekLinks(ratio='all',
                                     random_state=self.random_state,
                                     n_jobs=self.n_jobs)
        # If an object was given, affect
        elif self.tomek is not None:
            if isinstance(self.tomek, TomekLinks):
                self.tomek_ = self.tomek
            else:
                raise ValueError('tomek needs to be a TomekLinks object.'
                                 'Got {} instead.'.format(type(self.tomek)))
        # Otherwise create a default TomekLinks
        else:
            self.tomek_ = TomekLinks(ratio='all',
                                     random_state=self.random_state)

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        y = check_target_type(y)
        self.ratio_ = self.ratio
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)

        return self

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape \
(n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`

        """
        self._validate_estimator()

        X_res, y_res = self.smote_.fit_sample(X, y)
        return self.tomek_.fit_sample(X_res, y_res)
