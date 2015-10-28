UnbalancedDataset
=================

UnbalancedDataset is a python package offering a number of re-sampling techniques commonly used in datasets showing strong between-class imbalance.

[![Code Health](https://landscape.io/github/fmfn/UnbalancedDataset/master/landscape.svg?style=flat)](https://landscape.io/github/fmfn/UnbalancedDataset/master)

Installation
============

### Dependencies

* numpy
* scikit-learn

### Installation

UnbalancedDataset is not currently available on PyPi. To install the package, you will need to clone it and run the
setup.py file. Use the following commands to get a copy from Github and install all dependencies:

    git clone https://github.com/fmfn/UnbalancedDataset.git
    cd UnbalancedDataset
    python setup.py install

About
=====

Most classification algorithms will only perform optimally when the number of samples of each class is roughly the same. Highly skewed datasets, where the minority is heavily outnumbered by one or more classes, have proven to be a challenge while at the same time becoming more and more common.

One way of addresing this issue is by re-sampling the dataset as to offset this imbalance with the hope of arriving at a more robust and fair decision boundary than you would otherwise.

Re-sampling techniques are divided in two categories:
    1. Under-sampling the majority class(es).
    2. Over-sampling the minority class.
    
Bellow is a list of the methods currently implemented in this module.

* Under-sampling
    1. Random majority under-sampling with replacement
    2. Extraction of majority-minority Tomek links
    3. Under-sampling with Cluster Centroids
    4. NearMiss-(1 & 2 & 3)
    5. Condensend Nearest Neighbour
    6. One-Sided Selection
    7. Neighboorhood Cleaning Rule

* Over-sampling
    1. Random minority over-sampling with replacement
    2. SMOTE - Synthetic Minority Over-sampling Technique
    3. bSMOTE(1&2) - Borderline SMOTE of types 1 and 2
    4. SVM_SMOTE - Support Vectors SMOTE

* Over-sampling followed by under-sampling
    1. SMOTE + Tomek links
    2. SMOTE + ENN

* Ensemble sampling
    1. EasyEnsemble
    2. BalanceCascade

The different algorithms are presented in the [following notebook](https://github.com/glemaitre/UnbalancedDataset/blob/master/notebook/Notebook_UnbalancedDataset.ipynb).

This is a work in progress. Any comments, suggestions or corrections are welcome.

References:

1. NearMiss - ["kNN approach to unbalanced data distributions: A case study involving information extraction"](http://web0.site.uottawa.ca:4321/~nat/Workshop2003/jzhang.pdf), by Zhang et al., 2003.
1. CNN - ["Addressing the Curse of Imbalanced Training Sets: One-Sided Selection"](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/kubat97addressing.pdf), by Kubat et al., 1997.
1. One-Sided Selection - ["Addressing the Curse of Imbalanced Training Sets: One-Sided Selection"](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/kubat97addressing.pdf), by Kubat et al., 1997.
1. NCL - ["Improving identification of difficult small classes by balancing class distribution"](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/2001-Laurikkala-LNCS.pdf), by Laurikkala et al., 2001.
1. SMOTE - ["SMOTE: synthetic minority over-sampling technique"](https://www.jair.org/media/953/live-953-2037-jair.pdf), by Chawla et al., 2002.
1. Borderline SMOTE -  ["Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning"](http://sci2s.ugr.es/keel/keel-dataset/pdfs/2005-Han-LNCS.pdf), by Han et al., 2005
1. SVM_SMOTE - ["Borderline Over-sampling for Imbalanced Data Classification"](https://www.google.fr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0CDAQFjABahUKEwjH7qqamr_HAhWLthoKHUr0BIo&url=http%3A%2F%2Fousar.lib.okayama-u.ac.jp%2Ffile%2F19617%2FIWCIA2009_A1005.pdf&ei=a7zZVYeNDIvtasrok9AI&usg=AFQjCNHoQ6oC_dH1M1IncBP0ZAaKj8a8Cw&sig2=lh32CHGjs5WBqxa_l0ylbg), Nguyen et al., 2011.
1. SMOTE + Tomek - ["Balancing training data for automated annotation of keywords: a case study"](http://www.icmc.usp.br/~gbatista/files/wob2003.pdf), Batista et al., 2003.
1. SMOTE + ENN - ["A study of the behavior of several methods for balancing machine learning training data"](http://www.sigkdd.org/sites/default/files/issues/6-1-2004-06/batista.pdf), Batista et al., 2004.
1. EasyEnsemble & BalanceCascade - ["Exploratory Understanding for Class-Imbalance Learning"](http://cse.seu.edu.cn/people/xyliu/publication/tsmcb09.pdf), by Liu et al., 2009.
