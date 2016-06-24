UnbalancedDataset
=================

UnbalancedDataset is a python package offering a number of re-sampling techniques commonly used in datasets showing strong between-class imbalance.

[![Code Health](https://landscape.io/github/glemaitre/UnbalancedDataset/master/landscape.svg?style=flat)](https://landscape.io/github/glemaitre/UnbalancedDataset/master)
[![Build Status](https://travis-ci.org/glemaitre/UnbalancedDataset.svg?branch=master)](https://travis-ci.org/glemaitre/UnbalancedDataset)
[![Coverage Status](https://coveralls.io/repos/github/glemaitre/UnbalancedDataset/badge.svg?branch=master)](https://coveralls.io/github/glemaitre/UnbalancedDataset?branch=master)
[![Join the chat at https://gitter.im/glemaitre/UnbalancedDataset](https://badges.gitter.im/glemaitre/UnbalancedDataset.svg)](https://gitter.im/glemaitre/UnbalancedDataset?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Documentation
=============

Installation documentation, API documentation, and examples can be found on the [documentation](http://glemaitre.github.io/UnbalancedDataset)

Installation
============

### Dependencies

UnbalancedDataset is tested to work under Python 2.7 and Python 3.5.

* scipy(>=0.17.0)
* numpy(>=1.10.4)
* scikit-learn(>=0.17.1)

### Installation

UnbalancedDataset is not currently available on the PyPi's reporitories, 
however you can install it via `pip`:

    pip install git+https://github.com/fmfn/UnbalancedDataset

If you prefer, you can clone it and run the setup.py file. Use the following commands to get a 
copy from Github and install all dependencies:

    git clone https://github.com/fmfn/UnbalancedDataset.git
    cd UnbalancedDataset
    python setup.py install

### Testing

After installation, you can use `nose` to run the test suite:

```
make coverage
```

About
=====

Most classification algorithms will only perform optimally when the number of samples of each class is roughly the same. Highly skewed datasets, where the minority is heavily outnumbered by one or more classes, have proven to be a challenge while at the same time becoming more and more common.

One way of addresing this issue is by re-sampling the dataset as to offset this imbalance with the hope of arriving at a more robust and fair decision boundary than you would otherwise.

Re-sampling techniques are divided in two categories:
    1. Under-sampling the majority class(es).
    2. Over-sampling the minority class.
	3. Combining over- and under-sampling.
	4. Create ensemble balanced sets.
    
Bellow is a list of the methods currently implemented in this module.

* Under-sampling
    1. Random majority under-sampling with replacement
    2. [Extraction of majority-minority Tomek links](#ref1)
    3. Under-sampling with Cluster Centroids
    4. [NearMiss-(1 & 2 & 3)](#ref2)
    5. [Condensend Nearest Neighbour](#ref3)
    6. [One-Sided Selection](#ref4)
    7. [Neighboorhood Cleaning Rule](#ref5)
	8. [Edited Nearest Neighbours](#ref6)
	9. [Instance Hardness Threshold](#ref7)

* Over-sampling
    1. Random minority over-sampling with replacement
    2. [SMOTE - Synthetic Minority Over-sampling Technique](#ref8)
    3. [bSMOTE(1 & 2) - Borderline SMOTE of types 1 and 2](#ref9)
    4. [SVM SMOTE - Support Vectors SMOTE](#ref10)

* Over-sampling followed by under-sampling
    1. [SMOTE + Tomek links](#ref12)
    2. [SMOTE + ENN](#ref11)

* Ensemble sampling
    1. [EasyEnsemble](#ref13)
    2. [BalanceCascade](#ref13)

The different algorithms are presented in the [following notebook](https://github.com/fmfn/UnbalancedDataset/blob/master/examples/plot_unbalanced_dataset.ipynb).

This is a work in progress. Any comments, suggestions or corrections are welcome.

References:
-----------

<a name="ref1"></a>[1]: I. Tomek, [“Two modifications of CNN,”](http://sci2s.ugr.es/keel/pdf/algorithm/articulo/1976-Tomek-IEEETSMC(2).pdf) In Systems, Man, and Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 2010.

<a name="ref2"></a>[2]: I. Mani, I. Zhang. [“kNN approach to unbalanced data distributions: a case study involving information extraction,”](http://web0.site.uottawa.ca:4321/~nat/Workshop2003/jzhang.pdf) In Proceedings of workshop on learning from imbalanced datasets, 2003.

<a name="ref3"></a>[3]: P. Hart, [“The condensed nearest neighbor rule,”](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=1054155&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D1054155) In Information Theory, IEEE Transactions on, vol. 14(3), pp. 515-516, 1968.

<a name="ref4"></a>[4]: M. Kubat, S. Matwin, [“Addressing the curse of imbalanced training sets: one-sided selection,”](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/kubat97addressing.pdf) In ICML, vol. 97, pp. 179-186, 1997.

<a name="ref5"></a>[5]: J. Laurikkala, [“Improving identification of difficult small classes by balancing class distribution,”](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/2001-Laurikkala-LNCS.pdf) Springer Berlin Heidelberg, 2001.

<a name="ref6"></a>[6]: D. Wilson, [“Asymptotic Properties of Nearest Neighbor Rules Using Edited Data,”](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4309137&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D4309137) In IEEE Transactions on Systems, Man, and Cybernetrics, vol. 2 (3), pp. 408-421, 1972.

<a name="ref7"></a>[7]: D. Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier. [“An instance level analysis of data complexity.”](http://axon.cs.byu.edu/papers/smith.ml2013.pdf) Machine learning 95.2 (2014): 225-256.

<a name="ref8"></a>[8]: N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, [“SMOTE: synthetic minority over-sampling technique,”](https://www.jair.org/media/953/live-953-2037-jair.pdf) Journal of artificial intelligence research, 321-357, 2002.

<a name="ref9"></a>[9]: H. Han, W. Wen-Yuan, M. Bing-Huan, [“Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning,”](http://sci2s.ugr.es/keel/keel-dataset/pdfs/2005-Han-LNCS.pdf) Advances in intelligent computing, 878-887, 2005.

<a name="ref10"></a>[10]: H. M. Nguyen, E. W. Cooper, K. Kamei, [“Borderline over-sampling for imbalanced data classification,”](https://www.google.fr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0CDAQFjABahUKEwjH7qqamr_HAhWLthoKHUr0BIo&url=http%3A%2F%2Fousar.lib.okayama-u.ac.jp%2Ffile%2F19617%2FIWCIA2009_A1005.pdf&ei=a7zZVYeNDIvtasrok9AI&usg=AFQjCNHoQ6oC_dH1M1IncBP0ZAaKj8a8Cw&sig2=lh32CHGjs5WBqxa_l0ylbg) International Journal of Knowledge Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2001.

<a name="ref11"></a>[11]: G. Batista, R. C. Prati, M. C. Monard. [“A study of the behavior of several methods for balancing machine learning training data,”](http://www.sigkdd.org/sites/default/files/issues/6-1-2004-06/batista.pdf) ACM Sigkdd Explorations Newsletter 6 (1), 20-29, 2004.

<a name="ref12"></a>[12]: G. Batista, B. Bazzan, M. Monard, [“Balancing Training Data for Automated Annotation of Keywords: a Case Study,”](http://www.icmc.usp.br/~gbatista/files/wob2003.pdf) In WOB, 10-18, 2003.

<a name="ref13"></a>[13]: X. Y. Liu, J. Wu and Z. H. Zhou, [“Exploratory Undersampling for Class-Imbalance Learning,”](http://cse.seu.edu.cn/people/xyliu/publication/tsmcb09.pdf) in IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539-550, April 2009.
