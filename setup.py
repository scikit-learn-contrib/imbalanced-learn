#! /usr/bin/env python
"""Toolbox for imbalanced dataset in machine learning."""

import codecs
import os

from setuptools import find_packages, setup

try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.7 is needed.
    import __builtin__ as builtins

# This is a bit (!) hackish: we are setting a global variable so that the
# main imblearn __init__ can detect if it is being loaded by the setup
# routine, to avoid attempting to load components that aren't built yet:
# the numpy distutils extensions that are used by imbalanced-learn to
# recursively build the compiled extensions in sub-packages is based on the
# Python import machinery.
builtins.__IMBLEARN_SETUP__ = True

import imblearn._min_dependencies as min_deps  # noqa

# get __version__ from _version.py
ver_file = os.path.join("imblearn", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "imbalanced-learn"
DESCRIPTION = "Toolbox for imbalanced dataset in machine learning."
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "G. Lemaitre, C. Aridas"
MAINTAINER_EMAIL = "g.lemaitre58@gmail.com, ichkoar@gmail.com"
URL = "https://github.com/scikit-learn-contrib/imbalanced-learn"
LICENSE = "MIT"
DOWNLOAD_URL = "https://github.com/scikit-learn-contrib/imbalanced-learn"
VERSION = __version__  # noqa
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: C",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
PYTHON_REQUIRES = ">=3.8"
INSTALL_REQUIRES = (min_deps.tag_to_packages["install"],)
EXTRAS_REQUIRE = {
    key: value for key, value in min_deps.tag_to_packages.items() if key != "install"
}


setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
