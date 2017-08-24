"""Test for the deprecation helper"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from imblearn.utils.deprecation import deprecate_parameter
from imblearn.utils.testing import warns


class Sampler(object):
    def __init__(self):
        self.a = 'something'
        self.b = 'something'


def test_deprecate_parameter():
    with warns(DeprecationWarning, match="is deprecated from"):
        deprecate_parameter(Sampler(), '0.2', 'a')
    with warns(DeprecationWarning, match="Use 'b' instead."):
        deprecate_parameter(Sampler(), '0.2', 'a', 'b')
