"""Test the module easy ensemble."""
from __future__ import print_function

from os import remove
from os.path import join

from imblearn.datasets import fetch_benchmark
from sklearn.datasets.base import get_data_home


def test_fetch_data():
    """Testing that fetching the data is working."""

    # Download and extract the data
    data = fetch_benchmark()

    # Redo the same to check if this is working if the archive is
    # already existing
    data = fetch_benchmark()

    # Remove a file and check that the decompressing is working
    data_home = get_data_home(None)
    remove(join(data_home, 'x1data.npz'))
