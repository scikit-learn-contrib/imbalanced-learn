# This file is here so that when running from the root folder
# ./imblearn is added to sys.path by pytest.
# See https://docs.pytest.org/en/latest/pythonpath.html for more details.
# For example, this allows to build extensions in place and run pytest
# doc/modules/clustering.rst and use imblearn from the local folder
# rather than the one from site-packages.

import os

import numpy as np
import pytest
from sklearn.utils.fixes import parse_version

# use legacy numpy print options to avoid failures due to NumPy 2.+ scalar
# representation
if parse_version(np.__version__) > parse_version("2.0.0"):
    np.set_printoptions(legacy="1.25")


def pytest_runtest_setup(item):
    fname = item.fspath.strpath
    if (
        fname.endswith(os.path.join("keras", "_generator.py"))
        or fname.endswith(os.path.join("tensorflow", "_generator.py"))
        or fname.endswith("miscellaneous.rst")
    ):
        try:
            import tensorflow  # noqa
        except ImportError:
            pytest.skip("The tensorflow package is not installed.")
