"""Test the module SMOTENC."""
# Authors: Andrea Lorenzon <andrelorenzon@gmail.com>
# License: MIT

import pytest

import numpy as np

from imblearn.over_sampling import ROSE


def test_testunit():
    return True

def test_randomState():
    assert(np.random.RandomState(42))

def test_instance():
    rose = ROSE()
    assert(ROSE)

