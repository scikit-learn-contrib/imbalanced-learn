"""Test for the exceptions modules"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT


from imblearn.exceptions import raise_isinstance_error

from pytest import raises


def test_raise_isinstance_error():
    var = 10.0
    with raises(ValueError, match="has to be one of"):
        raise_isinstance_error('var', [int], var)
