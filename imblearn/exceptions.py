"""
The :mod:`imblearn.exceptions` module includes all custom warnings and error
classes used across imbalanced-learn.
"""

__all__ = ['SkipTestWarning']


class SkipTestWarning(UserWarning):
    """Warning class used to notify the user of a test that was skipped.

    For example, one of the estimator checks requires a pandas import.
    If the pandas package cannot be imported, the test will be skipped rather
    than register as a failure. Imported from scikit-learn.
    """
