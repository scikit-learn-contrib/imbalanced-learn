#! /usr/bin/env python
"""Toolbox for imbalanced dataset in machine learning."""

import codecs
import os
import sys

from setuptools import find_packages, setup


def load_version():
    """Executes imblearn/version.py in a globals dictionary and
    return it.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with codecs.open(os.path.join('imblearn', 'version.py'),
                     encoding='utf-8-sig') as fp:
        exec(fp.read(), globals_dict)

    return globals_dict


def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = set(['install', 'develop'])
    return install_commands.intersection(set(sys.argv))


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

descr = """Toolbox for imbalanced dataset in machine learning."""

_VERSION_GLOBALS = load_version()
DISTNAME = 'imbalanced-learn'
DESCRIPTION = 'Toolbox for imbalanced dataset in machine learning.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'G. Lemaitre, F. Nogueira, D. Oliveira, C. Aridas'
MAINTAINER_EMAIL = 'g.lemaitre58@gmail.com, fmfnogueira@gmail.com, dvro@cin.ufpe.br, char@upatras.gr'
URL = 'https://github.com/scikit-learn-contrib/imbalanced-learn'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/scikit-learn-contrib/imbalanced-learn'
VERSION = _VERSION_GLOBALS['__version__']


if __name__ == "__main__":
    if is_installing():
        module_check_fn = _VERSION_GLOBALS['_check_module_dependencies']
        module_check_fn(is_imbalanced_dataset_installing=True)

    install_requires = \
        ['%s>=%s' % (mod, meta['min_version'])
            for mod, meta in _VERSION_GLOBALS['REQUIRED_MODULE_METADATA']
            if not meta['required_at_installation']]

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
          classifiers=[
              'Intended Audience :: Science/Research',
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
          ],
          packages=find_packages(),
          install_requires=['scipy>=0.17.0',
                            'numpy>=1.10.4',
                            'scikit-learn>=0.17.1'])
