.. -*- mode: rst -*-

.. _scikit-learn: http://scikit-learn.org/stable/

.. _scikit-learn-contrib: https://github.com/scikit-learn-contrib

|Landscape|_ |Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |PythonVersion|_ |Pypi|_ |Gitter|_

.. |Landscape| image:: https://landscape.io/github/scikit-learn-contrib/imbalanced-learn/master/landscape.svg?style=flat
.. _Landscape: https://landscape.io/github/scikit-learn-contrib/imbalanced-learn/master

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/imbalanced-learn.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/imbalanced-learn

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/c8w4xb7re4euntvi/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/glemaitre/imbalanced-learn/history

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/imbalanced-learn/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/imbalanced-learn

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/imbalanced-learn.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/imbalanced-learn/tree/master

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/imbalanced-learn.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/imbalanced-learn.svg

.. |Pypi| image:: https://badge.fury.io/py/imbalanced-learn.svg
.. _Pypi: https://badge.fury.io/py/imbalanced-learn

.. |Gitter| image:: https://badges.gitter.im/scikit-learn-contrib/imbalanced-learn.svg
.. _Gitter: https://gitter.im/scikit-learn-contrib/imbalanced-learn?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

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

.. _documentation: http://contrib.scikit-learn.org/imbalanced-learn/

Installation
------------

Dependencies
~~~~~~~~~~~~

imbalanced-learn is tested to work under Python 2.7 and Python 3.5, and
3.6. The dependency requirements are based on the last scikit-learn release:

* scipy(>=0.13.3)
* numpy(>=1.8.2)
* scikit-learn(>=0.19.0)

Additionally, to run the examples, you need matplotlib(>=2.0.0).

Installation
~~~~~~~~~~~~

imbalanced-learn is currently available on the PyPi's repository and you can
install it via `pip`::

  pip install -U imbalanced-learn

The package is release also in Anaconda Cloud platform::

  conda install -c glemaitre imbalanced-learn

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies::

  git clone https://github.com/scikit-learn-contrib/imbalanced-learn.git
  cd imbalanced-learn
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/scikit-learn-contrib/imbalanced-learn.git

Testing
~~~~~~~

After installation, you can use `nose` to run the test suite::

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

Re-sampling techniques are divided in two categories:
    1. Under-sampling the majority class(es).
    2. Over-sampling the minority class.
    3. Combining over- and under-sampling.
    4. Create ensemble balanced sets.

Below is a list of the methods currently implemented in this module.

* Under-sampling
    1. Random majority under-sampling with replacement
    2. Extraction of majority-minority Tomek links [1]_
    3. Under-sampling with Cluster Centroids
    4. NearMiss-(1 & 2 & 3) [2]_
    5. Condensend Nearest Neighbour [3]_
    6. One-Sided Selection [4]_
    7. Neighboorhood Cleaning Rule [5]_
    8. Edited Nearest Neighbours [6]_
    9. Instance Hardness Threshold [7]_
    10. Repeated Edited Nearest Neighbours [14]_
    11. AllKNN [14]_

* Over-sampling
    1. Random minority over-sampling with replacement
    2. SMOTE - Synthetic Minority Over-sampling Technique [8]_
    3. bSMOTE(1 & 2) - Borderline SMOTE of types 1 and 2 [9]_
    4. SVM SMOTE - Support Vectors SMOTE [10]_
    5. ADASYN - Adaptive synthetic sampling approach for imbalanced learning [15]_

* Over-sampling followed by under-sampling
    1. SMOTE + Tomek links [12]_
    2. SMOTE + ENN [11]_

* Ensemble sampling
    1. EasyEnsemble [13]_
    2. BalanceCascade [13]_

The different algorithms are presented in the sphinx-gallery_.

.. _sphinx-gallery: http://contrib.scikit-learn.org/imbalanced-learn/auto_examples


References:
-----------

.. [1] : I. Tomek, “Two modifications of CNN,” In Systems, Man, and Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 2010.

.. [2] : I. Mani, I. Zhang. “kNN approach to unbalanced data distributions: a case study involving information extraction,” In Proceedings of workshop on learning from imbalanced datasets, 2003.

.. [3] : P. Hart, “The condensed nearest neighbor rule,” In Information Theory, IEEE Transactions on, vol. 14(3), pp. 515-516, 1968.

.. [4] : M. Kubat, S. Matwin, “Addressing the curse of imbalanced training sets: one-sided selection,” In ICML, vol. 97, pp. 179-186, 1997.

.. [5] : J. Laurikkala, “Improving identification of difficult small classes by balancing class distribution,” Springer Berlin Heidelberg, 2001.

.. [6] : D. Wilson, “Asymptotic Properties of Nearest Neighbor Rules Using Edited Data,” In IEEE Transactions on Systems, Man, and Cybernetrics, vol. 2 (3), pp. 408-421, 1972.

.. [7] : D. Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier. “An instance level analysis of data complexity.” Machine learning 95.2 (2014): 225-256.

.. [8] : N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of artificial intelligence research, 321-357, 2002.

.. [9] : H. Han, W. Wen-Yuan, M. Bing-Huan, “Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning,” Advances in intelligent computing, 878-887, 2005.

.. [10] : H. M. Nguyen, E. W. Cooper, K. Kamei, “Borderline over-sampling for imbalanced data classification,” International Journal of Knowledge Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2001.

.. [11] : G. Batista, R. C. Prati, M. C. Monard. “A study of the behavior of several methods for balancing machine learning training data,” ACM Sigkdd Explorations Newsletter 6 (1), 20-29, 2004.

.. [12] : G. Batista, B. Bazzan, M. Monard, [“Balancing Training Data for Automated Annotation of Keywords: a Case Study,” In WOB, 10-18, 2003.

.. [13] : X. Y. Liu, J. Wu and Z. H. Zhou, “Exploratory Undersampling for Class-Imbalance Learning,” in IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539-550, April 2009.

.. [14] : I. Tomek, “An Experiment with the Edited Nearest-Neighbor Rule,” IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6), pp. 448-452, June 1976.

.. [15] : He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. “ADASYN: Adaptive synthetic sampling approach for imbalanced learning,” In IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), pp. 1322-1328, 2008.
