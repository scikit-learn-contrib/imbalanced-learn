"""Class to perform over-sampling using SMOTE and cleaning using ENN."""
from __future__ import print_function
from __future__ import division

from sklearn.utils import check_X_y

from ..over_sampling import SMOTE
from ..under_sampling import EditedNearestNeighbours
from ..base import SamplerMixin


class SMOTEENN(SamplerMixin):
    """Class to perform over-sampling using SMOTE and cleaning using ENN.

    Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the
        number of samples in the minority class over the the number of
        samples in the majority class.

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Whether or not to print information about the processing.

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
        options: 'regular', 'borderline1', 'borderline2', 'svm'.

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
    ratio : str or float
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the
        number of samples in the minority class over the the number of
        samples in the majority class.

    random_state : int or None
        Seed for random number generation.

    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    Notes
    -----
    The method is presented in [1]_.

    This class does not support mutli-class.

    References
    ----------
    .. [1] G. Batista, R. C. Prati, M. C. Monard. "A study of the behavior of
       several methods for balancing machine learning training data," ACM
       Sigkdd Explorations Newsletter 6 (1), 20-29, 2004.

    """

    def __init__(self, ratio='auto', random_state=None, verbose=True,
                 k=5, m=10, out_step=0.5, kind_smote='regular',
                 size_ngh=3, kind_enn='all', n_jobs=-1, **kwargs):

        """Initialise the SMOTE ENN object.

        Parameters
        ----------
        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balance
            the dataset. Otherwise, the ratio is defined as the
            number of samples in the minority class over the the number of
            samples in the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Whether or not to print information about the processing.

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
            options: 'regular', 'borderline1', 'borderline2', 'svm'.

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

        Returns
        -------
        None

        """
        super(SMOTEENN, self).__init__(ratio=ratio, random_state=random_state,
                                       verbose=verbose)

        self.k = k
        self.m = m
        self.out_step = out_step
        self.kind_smote = kind_smote
        self.n_jobs = n_jobs
        self.kwargs = kwargs

        self.sm = SMOTE(ratio=self.ratio, random_state=self.random_state,
                        verbose=self.verbose, k=self.k, m=self.m,
                        out_step=self.out_step, kind=self.kind_smote,
                        n_jobs=self.n_jobs, **self.kwargs)

        self.size_ngh = size_ngh
        self.kind_enn = kind_enn

        self.enn = EditedNearestNeighbours(random_state=self.random_state,
                                           verbose=self.verbose,
                                           size_ngh=self.size_ngh,
                                           kind_sel=self.kind_enn,
                                           n_jobs=self.n_jobs)

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
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        super(SMOTEENN, self).fit(X, y)

        # Fit using SMOTE
        self.sm.fit(X, y)

        return self

    def sample(self, X, y):
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
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        super(SMOTEENN, self).sample(X, y)

        # Transform using SMOTE
        X, y = self.sm.sample(X, y)

        # Fit and transform using ENN
        return self.enn.fit_sample(X, y)
