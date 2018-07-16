#! /usr/bin/env python
"""Toolbox for imbalanced dataset in machine learning."""

import codecs
import os
import subprocess
import sys
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
TREE_SPLIT_PACKAGE = 'imblearn/tree_split'
CHECK_BUILD_PACKAGE = 'imblearn/__check_build'
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn']
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
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6']

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage(CHECK_BUILD_PACKAGE)
    config.add_subpackage(TREE_SPLIT_PACKAGE)

    return config

def generate_cython(package):
    """Cythonize all sources in the package"""
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd,
                                      'build_tools/cython',
                                      'cythonize.py'),
                         package],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def setup_package():
    from numpy.distutils.core import setup

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    src_path = local_path

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        generate_cython(CHECK_BUILD_PACKAGE)
        generate_cython(TREE_SPLIT_PACKAGE)

    try:
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
			  configuration=configuration,
              packages=find_packages(),
              install_requires=INSTALL_REQUIRES)
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return


if __name__ == '__main__':
    setup_package()
