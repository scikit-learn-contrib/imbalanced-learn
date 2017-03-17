from sklearn.utils.testing import assert_raises_regex

from imblearn.exceptions import raise_isinstance_error


def test_raise_isinstance_error():
    var = 10.0
    assert_raises_regex(ValueError, "has to be one of",
                        raise_isinstance_error, 'var', [int], var)
