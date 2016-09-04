"""Imbalanced dataset for benchmarking.

The datasets composing the benchmark are available at:
https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz

It is a compilation of UCI, LIBSVM, and KDD datasets.
"""

import tarfile
import logging

from os import makedirs
from os.path import join
from os.path import exists

try:
    # Python 2
    import urllib2
    urlopen = urllib2.urlopen
except ImportError:
    # Python 3
    import urllib.request
    urlopen = urllib.request.urlopen

import numpy as np

from sklearn.datasets.base import get_data_home


DATA_URL = 'https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz'
ARCHIVE_NAME = "benchmark-imbalanced-learn.tar.gz"
TARGET_FILENAME = ['x{}data.npz'.format(idx+1) for idx in range(27)]

logger = logging.getLogger(__name__)

# Grab the module-level docstring to use as a description of the
# dataset
MODULE_DOCS = __doc__


def fetch_benchmark(data_home=None, download_if_missing=True):
    """Loader for the imbalanced dataset used as benchmark.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing: optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------

    data : list of npz object
        List of object containing the information about each dataset.
        Each object in the list is composed of 2 numpy arrays:
        - `data` : ndarray, shape (n_samples, n_features) containing the data,
        - `label` : ndarray, shape (n_samples, ) containing the label
        associated to data.

    """

    # Check that the data folder is existing
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)

    # Create the filename to work with
    filepath = [join(data_home, f) for f in TARGET_FILENAME]
    archive_path = join(data_home, ARCHIVE_NAME)
    # Check that these files are all existing, otherwise we will download them
    b_download = False
    b_extract = False
    for f in filepath:
        if not exists(f):
            b_download = True
            b_extract = True
    if exists(archive_path):
        b_download = False
        b_extract = True

    # Check if we need to download the data
    if b_download:
        if download_if_missing:
            logger.info('Download the benchmark dataset')
            opener = urlopen(DATA_URL)
            with open(archive_path, 'wb') as f:
                f.write(opener.read())
        else:
            raise IOError('Benchmark dataset not found')
    else:
        logger.info('The dataset was already downloaded')

    # Check if we need to extract the data
    if b_extract:
        logger.info("Decompressing %s", archive_path)
        tarfile.open(archive_path, "r:gz").extractall(path=data_home)
    else:
        logger.info('No need to extract the data, they are already'
                         ' available')

    # Finally load the data
    return [np.load(f) for f in filepath]
