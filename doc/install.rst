########################
Install and contribution
########################

Prerequisites
=============

The imbalanced-learn package requires the following dependencies:

* numpy (>=1.8.2)
* scipy (>=0.13.3)
* scikit-learn (>=0.20)
* keras 2 (optional)
* tensorflow (optional)

Our release policy is to follow the scikit-learn releases in order to
synchronize the new feature. **imbalanced-learn 0.4 is the last version to
support Python 2.7**

Install
=======

imbalanced-learn is currently available on the PyPi's reporitories and you can
install it via `pip`::

  pip install -U imbalanced-learn

The package is release also in Anaconda Cloud platform::

  conda install -c conda-forge imbalanced-learn

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all dependencies::

  git clone https://github.com/scikit-learn-contrib/imbalanced-learn.git
  cd imbalanced-learn
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/scikit-learn-contrib/imbalanced-learn.git

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
