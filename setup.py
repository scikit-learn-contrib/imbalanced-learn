#! /usr/bin/env python
"""Toolbox for imbalanced dataset in machine learning."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('imblearn', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'imbalanced-learn'
DESCRIPTION = 'Toolbox for imbalanced dataset in machine learning.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'G. Lemaitre, C. Aridas'
MAINTAINER_EMAIL = 'g.lemaitre58@gmail.com, ichkoar@gmail.com'
URL = 'https://github.com/scikit-learn-contrib/imbalanced-learn'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/scikit-learn-contrib/imbalanced-learn'
VERSION = __version__
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: C',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
INSTALL_REQUIRES = ['numpy>=1.8.2', 'scipy>=0.13.3', 'scikit-learn>=0.20']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib',
        'pandas',
        'keras',
        'tensorflow'
    ]
}


setup(name=DISTNAME,
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
      extras_require=EXTRAS_REQUIRE)
