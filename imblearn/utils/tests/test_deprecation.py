"""Test for the deprecation helper"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from sklearn.utils.testing import assert_warns_message

from imblearn.utils.deprecation import deprecate_parameter


class Sampler(object):
    def __init__(self):
        self.a = 'something'
        self.b = 'something'


def test_deprecate_parameter():
    assert_warns_message(DeprecationWarning, "is deprecated from",
                         deprecate_parameter, Sampler(), '0.2', 'a')
    assert_warns_message(DeprecationWarning, "Use 'b' instead.",
                         deprecate_parameter, Sampler(), '0.2', 'a', 'b')
