.. -*- mode: rst -*-

.. _scikit-learn: http://scikit-learn.org/stable/

.. _scikit-learn-contrib: https://github.com/scikit-learn-contrib

|Azure|_ |Codecov|_ |CircleCI|_ |PythonVersion|_ |Pypi|_ |Gitter|_ |Black|_

.. |Azure| image:: https://dev.azure.com/imbalanced-learn/imbalanced-learn/_apis/build/status/scikit-learn-contrib.imbalanced-learn?branchName=master
.. _Azure: https://dev.azure.com/imbalanced-learn/imbalanced-learn/_build

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/imbalanced-learn/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/imbalanced-learn

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/imbalanced-learn.svg?style=shield
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/imbalanced-learn/tree/master

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/imbalanced-learn.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/imbalanced-learn.svg

.. |Pypi| image:: https://badge.fury.io/py/imbalanced-learn.svg
.. _Pypi: https://badge.fury.io/py/imbalanced-learn

.. |Gitter| image:: https://badges.gitter.im/scikit-learn-contrib/imbalanced-learn.svg
.. _Gitter: https://gitter.im/scikit-learn-contrib/imbalanced-learn?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: :target: https://github.com/psf/black

.. |PythonMinVersion| replace:: 3.8
.. |NumPyMinVersion| replace:: 1.17.3
.. |SciPyMinVersion| replace:: 1.5.0
.. |ScikitLearnMinVersion| replace:: 1.0.2
.. |MatplotlibMinVersion| replace:: 3.1.2
.. |PandasMinVersion| replace:: 1.0.5
.. |TensorflowMinVersion| replace:: 2.4.3
.. |KerasMinVersion| replace:: 2.4.3
.. |SeabornMinVersion| replace:: 0.9.0
.. |PytestMinVersion| replace:: 5.0.1

imbalanced-learn
================

imbalanced-learn is a python package offering a number of re-sampling techniques
commonly used in datasets showing strong between-class imbalance.
It is compatible with scikit-learn_ and is part of scikit-learn-contrib_
projects.

Documentation
-------------

Installation documentation, API documentation, and examples can be found on the
documentation_.

.. _documentation: https://imbalanced-learn.org/stable/

Installation
------------

Dependencies
~~~~~~~~~~~~

`imbalanced-learn` requires the following dependencies:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- Scikit-learn (>= |ScikitLearnMinVersion|)

Additionally, `imbalanced-learn` requires the following optional dependencies:

- Pandas (>= |PandasMinVersion|) for dealing with dataframes
- Tensorflow (>= |TensorflowMinVersion|) for dealing with TensorFlow models
- Keras (>= |KerasMinVersion|) for dealing with Keras models

The examples will requires the following additional dependencies:

- Matplotlib (>= |MatplotlibMinVersion|)
- Seaborn (>= |SeabornMinVersion|)

Installation
~~~~~~~~~~~~

From PyPi or conda-forge repositories
.....................................

imbalanced-learn is currently available on the PyPi's repositories and you can
install it via `pip`::

  pip install -U imbalanced-learn

The package is release also in Anaconda Cloud platform::

  conda install -c conda-forge imbalanced-learn

From source available on GitHub
...............................

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all dependencies::

  git clone https://github.com/scikit-learn-contrib/imbalanced-learn.git
  cd imbalanced-learn
  pip install .

Be aware that you can install in developer mode with::

  pip install --no-build-isolation --editable .

If you wish to make pull-requests on GitHub, we advise you to install
pre-commit::

  pip install pre-commit
  pre-commit install

Testing
~~~~~~~

After installation, you can use `pytest` to run the test suite::

  make coverage

Development
-----------

The development of this scikit-learn-contrib is in line with the one
of the scikit-learn community. Therefore, you can refer to their
`Development Guide
<http://scikit-learn.org/stable/developers>`_.

About
-----

If you use imbalanced-learn in a scientific publication, we would appreciate
citations to the following paper::

  @article{JMLR:v18:16-365,
  author  = {Guillaume  Lema{{\^i}}tre and Fernando Nogueira and Christos K. Aridas},
  title   = {Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2017},
  volume  = {18},
  number  = {17},
  pages   = {1-5},
  url     = {http://jmlr.org/papers/v18/16-365}
  }

Most classification algorithms will only perform optimally when the number of
samples of each class is roughly the same. Highly skewed datasets, where the
minority is heavily outnumbered by one or more classes, have proven to be a
challenge while at the same time becoming more and more common.

One way of addressing this issue is by re-sampling the dataset as to offset this
imbalance with the hope of arriving at a more robust and fair decision boundary
than you would otherwise.

You can refer to the `imbalanced-learn`_ documentation to find details about
the implemented algorithms.

.. _imbalanced-learn: https://imbalanced-learn.org/stable/user_guide.html
