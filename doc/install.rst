.. _getting_started:

###############
Getting Started
###############

Prerequisites
=============

You can find the complete list of the dependencies in the following table:

.. include:: min_dependency_table.rst

Install
=======

From PyPi or conda-forge repositories
-------------------------------------

imbalanced-learn is currently available on the PyPi's repositories and you can
install it via `pip`::

  pip install imbalanced-learn

The package is released also on the conda-forge repositories and you can install
it with `conda` (or `mamba`)::

  conda install -c conda-forge imbalanced-learn

Intel optimizations via scikit-learn-intelex
--------------------------------------------

Imbalanced-learn relies entirely on scikit-learn algorithms. Intel provides an
optimized version of scikit-learn for Intel hardwares, called scikit-learn-intelex.
Installing scikit-learn-intelex and patching scikit-learn will activate the
Intel optimizations.

You can refer to the following
`blog post <https://medium.com/intel-analytics-software/why-pay-more-for-machine-learning-893683bd78e4>`_
for some benchmarks.

Refer to the following documentation for instructions:

- `Installation guide <https://intel.github.io/scikit-learn-intelex/installation.html>`_.
- `Patching guide <https://intel.github.io/scikit-learn-intelex/what-is-patching.html>`_.

From source available on GitHub
-------------------------------

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

Test and coverage
=================

You want to test the code before to install::

  $ make test

You wish to test the coverage of your version::

  $ make coverage

You can also use `pytest`::

  $ pytest imblearn -v

Contribute
==========

You can contribute to this code through Pull Request on GitHub_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitHub: https://github.com/scikit-learn-contrib/imbalanced-learn/pulls
