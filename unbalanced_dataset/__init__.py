"""Toolbox for unbalanced dataset in machine learning.

``UnbalancedDataset`` is a set of python methods to deal with unbalanced
datset in machine learning and pattern recognition.

Subpackages
-----------
combine
    Module which provides methods based on over-sampling and under-sampling.
ensemble
    Module which provides methods generating an ensemble of
    under-sampled subsets.
over_sampling
    Module which provides methods to under-sample a dataset.
under-sampling
    Module which provides methods to over-sample a dataset.
utils
    Module which provides helper methods.
"""

import os
import imp
import functools
import warnings
import sys

__version__ = '0.1.dev0'

pkg_dir = os.path.abspath(os.path.dirname(__file__))

# Logic for checking for improper install and importing while in the source
# tree when package has not been installed inplace.
# Code adapted from scikit-learn's __check_build module.
_INPLACE_MSG = """
It appears that you are importing a local UnbalancedDataset source tree. For
this, you need to have an inplace install. Maybe you are in the source
directory and you need to try from another location."""

_STANDARD_MSG = """
Your install of UnbalancedDataset package appears to be broken. """


def _raise_build_error(e):
    import os.path as osp
    # Raise a comprehensible error
    local_dir = osp.split(__file__)[0]
    msg = _STANDARD_MSG
    if local_dir == "unbalanced_dataset":
        # Picking up the local install: this will work only if
        # the install is an 'inplace build'
        msg = _INPLACE_MSG
    raise ImportError("""%s
    It seems that UnbalncedDataset has not been built correctly.
    %s""" % (e, msg))

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __UNBALANCED_DATASET_SETUP__
except NameError:
    __UNBALANCED_DATASET_SETUP__ = False

if __UNBALANCED_DATASET_SETUP__:
    sys.stderr.write('Partial import of UnbalancedDataset during the build'
                     ' process.\n')
    # We are not importing the rest of the scikit during the build
    # process, as it may not be compiled yet
else:
    __all__ = ['combine',
               'ensemble',
               'over_sampling',
               'under_sampling',
               'utils']

del warnings, functools, os, imp, sys
