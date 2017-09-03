"""Test for miscellaneous samplers objects."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from imblearn.misc import FunctionSampler


def function_sampler_identity():
    sampler = FunctionSampler(1)
