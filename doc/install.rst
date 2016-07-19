###############
Getting Started
###############

Install
=======

imbalanced-learn is currently available on the PyPi's reporitories and you can install it via `pip`::

  pip install -U imbalanced-learn

The package is release also in Anaconda Cloud platform::

  conda install -c glemaitre imbalanced-learn

If you prefer, you can clone it and run the setup.py file. Use the following commands to get a 
copy from Github and install all dependencies::

  git clone https://github.com/scikit-learn-contrib/imbalanced-learn.git
  cd imbalanced-learn
  python setup.py install

Test and coverage
=================

You want to test the code before to install::

  $ make test

You wish to test the coverage of your version::

  $ make coverage

Contribute
==========

You can contribute to this code through Pull Request on GitHub_. Please, make sure that your code is coming with unit tests to ensure full coverage and continuous integration in the API.

.. _GitHub: https://github.com/scikit-learn-contrib/imbalanced-learn/pulls
