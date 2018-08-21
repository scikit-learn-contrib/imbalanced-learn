"""Class to perform over-sampling using SMOTE and cleaning using ENN."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import logging
import warnings

from sklearn.utils import check_X_y

from ..base import SamplerMixin
from ..over_sampling import SMOTE
from ..over_sampling.base import BaseOverSampler
from ..under_sampling import EditedNearestNeighbours
from ..utils import check_target_type, hash_X_y
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class SMOTEENN(SamplerMixin):
    """Class to perform over-sampling using SMOTE and cleaning using ENN.

    Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.

    Read more in the :ref:`User Guide <combine>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    smote : object, optional (default=SMOTE())
        The :class:`imblearn.over_sampling.SMOTE` object to use. If not given,
        a :class:`imblearn.over_sampling.SMOTE` object with default parameters
        will be given.

    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.

    Notes
    -----
    The method is presented in [1]_.

    Supports multi-class resampling. Refer to SMOTE and ENN regarding the
    scheme which used.

    See :ref:`sphx_glr_auto_examples_combine_plot_smote_enn.py` and
    :ref:`sphx_glr_auto_examples_combine_plot_comparison_combine.py`.

    See also
    --------
    SMOTETomek : Over-sample using SMOTE followed by under-sampling removing
        the Tomek's links.

    References
    ----------
    .. [1] G. Batista, R. C. Prati, M. C. Monard. "A study of the behavior of
       several methods for balancing machine learning training data," ACM
       Sigkdd Explorations Newsletter 6 (1), 20-29, 2004.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.combine import SMOTEENN # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> sme = SMOTEENN(random_state=42)
    >>> X_res, y_res = sme.fit_sample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 881}})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 smote=None,
                 enn=None,
                 ratio=None):
        super(SMOTEENN, self).__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = smote
        self.enn = enn
        self.ratio = ratio
        self.logger = logging.getLogger(__name__)

    def _validate_estimator(self):
        "Private function to validate SMOTE and ENN objects"
        if self.smote is not None:
            if isinstance(self.smote, SMOTE):
                self.smote_ = self.smote
            else:
                raise ValueError('smote needs to be a SMOTE object.'
                                 'Got {} instead.'.format(type(self.smote)))
        # Otherwise create a default SMOTE
        else:
            self.smote_ = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                ratio=self.ratio)

        if self.enn is not None:
            if isinstance(self.enn, EditedNearestNeighbours):
                self.enn_ = self.enn
            else:
                raise ValueError('enn needs to be an EditedNearestNeighbours.'
                                 ' Got {} instead.'.format(type(self.enn)))
        # Otherwise create a default EditedNearestNeighbours
        else:
            self.enn_ = EditedNearestNeighbours(sampling_strategy='all')

    @property
    def ratio_(self):
        # FIXME: remove in 0.6
        warnings.warn("'ratio' and 'ratio_' are deprecated. Use "
                      "'sampling_strategy' and 'sampling_strategy_' instead.",
                      DeprecationWarning)
        return self.sampling_strategy_

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
        y = check_target_type(y)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        self.sampling_strategy_ = self.sampling_strategy
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

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """
        self._validate_estimator()

        X_res, y_res = self.smote_.fit_sample(X, y)
        return self.enn_.fit_sample(X_res, y_res)
