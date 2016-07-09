"""Class to perform over-sampling using SMOTE and cleaning using ENN."""
from __future__ import print_function
from __future__ import division

from ..over_sampling import SMOTE
from ..under_sampling import TomekLinks
from ..base import SamplerMixin


class SMOTETomek(SamplerMixin):
    """Class to perform over-sampling using SMOTE and cleaning using
    Tomek links.

    Combine over- and under-sampling using SMOTE and Tomek links.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the
        number of samples in the minority class over the the number of
        samples in the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    k : int, optional (default=5)
        Number of nearest neighbours to used to construct synthetic
        samples.

    m : int, optional (default=10)
        Number of nearest neighbours to use to determine if a minority
        sample is in danger.

    out_step : float, optional (default=0.5)
        Step size when extrapolating.

    kind_smote : str, optional (default='regular')
        The type of SMOTE algorithm to use one of the following
        options: 'regular', 'borderline1', 'borderline2', 'svm'

    size_ngh : int, optional (default=3)
        Size of the neighbourhood to consider to compute the average
        distance to the minority point samples.

    kind_sel : str, optional (default='all')
        Strategy to use in order to exclude samples.

        - If 'all', all neighbours will have to agree with the samples of
        interest to not be excluded.
        - If 'mode', the majority vote of the neighbours will be used in
        order to exclude a sample.

    n_jobs : int, optional (default=-1)
        The number of threads to open if possible.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    The methos is presented in [1]_.

    This class does not support mutli-class.

    References
    ----------
    .. [1] G. Batista, B. Bazzan, M. Monard, "Balancing Training Data for
       Automated Annotation of Keywords: a Case Study," In WOB, 10-18, 2003.

    """

    def __init__(self, ratio='auto', random_state=None,
                 k=5, m=10, out_step=0.5, kind_smote='regular',
                 n_jobs=-1, **kwargs):
        super(SMOTETomek, self).__init__(ratio=ratio)
        self.random_state = random_state
        self.k = k
        self.m = m
        self.out_step = out_step
        self.kind_smote = kind_smote
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.sm = SMOTE(ratio=self.ratio, random_state=self.random_state,
                        k=self.k, m=self.m, out_step=self.out_step,
                        kind=self.kind_smote, n_jobs=self.n_jobs,
                        **self.kwargs)
        self.tomek = TomekLinks(random_state=self.random_state)

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """

        super(SMOTETomek, self).fit(X, y)

        # Fit using SMOTE
        self.sm.fit(X, y)

        return self

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """

        # Transform using SMOTE
        X, y = self.sm.sample(X, y)

        # Fit and transform using ENN
        return self.tomek.fit_sample(X, y)
