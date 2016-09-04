"""Test the module easy ensemble."""
from __future__ import print_function

from os import remove
from os.path import join

from numpy.testing import assert_equal
from nose import SkipTest

from imblearn.datasets import fetch_benchmark


def test_fetch_data():
    """Testing that fetching the data is working."""

    # Download and extract the data
    try:
        data = fetch_benchmark(download_if_missing=False)
    except IOError:
        raise SkipTest("Download 20 newsgroups to run this test")

    # Check that we have the 27 dataset
    assert_equal(len(data), 27)

    # Check that each object has a 'data' and 'label' ndarray
    obj_name = ('data', 'label')
    for dataset in data:
        assert_equal(tuple([key for key in dataset.keys()]), obj_name)
