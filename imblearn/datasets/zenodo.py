"""Collection of imbalanced datasets.

This collection of datasets have been proposed in [1]_. The
characteristics of the available datasets are presented in the table
below.

|ID   |Name          |Repository & Target          |Ratio|#S     |#F |
|:---:|:------------:|-----------------------------|:---:|:-----:|:-:|
|1    |ecoli         |UCI, target: imU             |8.6:1|336    |7  |
|2    |optical_digits|UCI, target: 8               |9.1:1|5,620  |64 |
|3    |satimage      |UCI, target: 4               |9.3:1|6,435  |36 |
|4    |pen_digits    |UCI, target: 5               |9.4:1|10,992 |16 |
|5    |abalone       |UCI, target: 7               |9.7:1|4,177  |8  |
|6    |sick_euthyroid|UCI, target: sick euthyroid  |9.8:1|3,163  |25 |
|7    |spectrometer  |UCI, target: >=44            |11:1 |531    |93 |
|8    |car_eval_34   |UCI, target: good, v good    |12:1 |1,728  |6  |
|9    |isolet        |UCI, target: A, B            |12:1 |7,797  |617|
|10   |us_crime      |UCI, target: >0.65           |12:1 |1,994  |122|
|11   |yeast_ml8     |LIBSVM, target: 8            |13:1 |2,417  |103|
|12   |scene         |LIBSVM, target: >one label   |13:1 |2,407  |294|
|13   |libras_move   |UCI, target: 1               |14:1 |360    |90 |
|14   |thyroid_sick  |UCI, target: sick            |15:1 |3,772  |28 |
|15   |coil_2000     |KDD, CoIL, target: minority  |16:1 |9,822  |85 |
|16   |arrhythmia    |UCI, target: 06              |17:1 |452    |279|
|17   |solar_flare_m0|UCI, target: M->0            |19:1 |1,389  |10 |
|18   |oil           |UCI, target: minority        |22:1 |937    |49 |
|19   |car_eval_4    |UCI, target: vgood           |26:1 |1,728  |6  |
|20   |wine_quality  |UCI, wine, target: <=4       |26:1 |4,898  |11 |
|21   |letter_img    |UCI, target: Z               |26:1 |20,000 |16 |
|22   |yeast_me2     |UCI, target: ME2             |28:1 |1,484  |8  |
|23   |webpage       |LIBSVM, w7a, target: minority|33:1 |49,749 |300|
|24   |ozone_level   |UCI, ozone, data             |34:1 |2,536  |72 |
|25   |mammography   |UCI, target: minority        |42:1 |11,183 |6  |
|26   |protein_homo  |KDD CUP 2004, minority       |111:1|145,751|74 |
|27   |abalone_19    |UCI, target: 19              |130:1|4,177  |8  |

References
----------
.. [1] Ding, Zejin, "Diversified Ensemble Classifiers for Highly
   Imbalanced Data Learning and their Application in Bioinformatics."
   Dissertation, Georgia State University, (2011).

"""

# Author: Guillaume Lemaitre
# License: BSD 3 clause

from collections import OrderedDict
import tarfile
from io import BytesIO
import logging
from os.path import join, isfile
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

import numpy as np

from sklearn.datasets import get_data_home
from sklearn.datasets.base import Bunch
from sklearn.utils.fixes import makedirs
from sklearn.externals import six
from sklearn.utils import check_random_state

URL = ('https://zenodo.org/record/61452/files/'
       'benchmark-imbalanced-learn.tar.gz')
PRE_FILENAME = 'x'
POST_FILENAME = 'data.npz'

MAP_NAME_ID = OrderedDict({'ecoli': 1,
                           'optical_digits': 2,
                           'satimage': 3,
                           'pen_digits': 4,
                           'abalone': 5,
                           'sick_euthyroid': 6,
                           'spectrometer': 7,
                           'car_eval_34': 8,
                           'isolet': 9,
                           'us_crime': 10,
                           'yeast_ml8': 11,
                           'scene': 12,
                           'libras_move': 13,
                           'thyroid_sick': 14,
                           'coil_2000': 15,
                           'arrhythmia': 16,
                           'solar_flare_m0': 17,
                           'oil': 18,
                           'car_eval_4': 19,
                           'wine_quality': 20,
                           'letter_img': 21,
                           'yeast_me2': 22,
                           'webpage': 23,
                           'ozone_level': 24,
                           'mammography': 25,
                           'protein_homo': 26,
                           'abalone_19': 27})

logger = logging.getLogger()


def fetch_zenodo(data_home=None,
                 filter_data=None,
                 download_if_missing=True,
                 random_state=None,
                 shuffle=False):
    """Load the Higgs dataset, downloading it if necessary.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    data_home : string, optional (default=None)
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/scikit_learn_data' subfolders.

    filter_data : tuple of str/int or None, optional (default=None)
        A tuple containing the ID or the name of the datasets to be returned.
        Refer to the above table to get the ID and name of the datasets.

    download_if_missing : boolean, optional (default=True)
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None, optional (default=None)
        Random state for shuffling the dataset.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : bool, optional (default=False)
        Whether to shuffle dataset.

    Returns
    -------
    datasets : OrderedDict of Bunch object,
        The ordered is defined by ``filter_data``. Each Bunch object ---
        refered as dataset --- have the following attributes:

    dataset.data : ndarray, shape (n_samples, n_features)

    dataset.target : ndarray, shape (n_samples, )

    dataset.DESCR : string
        Description of the each dataset.

    """

    data_home = get_data_home(data_home=data_home)
    zenodo_dir = join(data_home, "zenodo")
    datasets = {}

    if filter_data is None:
        filter_data_ = MAP_NAME_ID.keys()
    else:
        list_data = MAP_NAME_ID.keys()
        filter_data_ = []
        for it in filter_data:
            if isinstance(it, six.string_types):
                if it not in list_data:
                    raise ValueError('{} is not a dataset available. '
                                     'The available datasets are {}'.format(
                                         it, list_data))
                else:
                    filter_data_.append(it)
            elif isinstance(it, int):
                if it < 1 or it > 27:
                    raise ValueError('The dataset with the ID={} is not an '
                                     'available dataset. The IDs are '
                                     '{}'.format(it, range(1, 28)))
                else:
                    # The index start at one, then we need to remove one
                    # to not have issue with the indexing.
                    filter_data_.append(MAP_NAME_ID.items()[it - 1])
            else:
                raise ValueError('The value in the should be str or int.'
                                 ' Got {} instead.'.format(type(it)))

    # go through the list and check if the data are available
    for it in filter_data_:
        filename = PRE_FILENAME + str(MAP_NAME_ID[it]) + POST_FILENAME
        filename = join(zenodo_dir, filename)
        available = isfile(filename)

        if download_if_missing and not available:
            makedirs(zenodo_dir, exist_ok=True)
            logger.warning("Downloading %s" % URL)
            f = BytesIO(urlopen(URL).read())
            tar = tarfile.open(fileobj=f)
            tar.extractall(path=zenodo_dir)
        elif not download_if_missing and not available:
            raise RuntimeError("The datasets are not available locally. Put "
                               "'download_if_missing' if you want to use the "
                               "data.")

        data = np.load(filename)
        X, y = data['data'], data['label']

        if shuffle:
            ind = np.arange(X.shape[0])
            rng = check_random_state(random_state)
            rng.shuffle(ind)
            X = X[ind]
            y = y[ind]

        datasets[it] = Bunch(data=X, target=y, DESCR=it)

    return datasets
