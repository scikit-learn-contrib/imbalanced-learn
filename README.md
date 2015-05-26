[![Code Health](https://landscape.io/github/fmfn/UnbalancedDataset/master/landscape.svg?style=flat)](https://landscape.io/github/fmfn/UnbalancedDataset/master)

UnbalancedDataset
=================

UnbalancedDataset is a python module offering a number of resampling techniques commonly used in datasets showing strong between-class imbalance.

Most classification algorithms will only perform optimally when the number of samples of each class is roughly the same. Highly skewed datasets, where the minority heavily outnumbered by one or more classes, haven proven to be a challenge while at the same time becoming more and more common.

One way of addresing this issue is by resampling the dataset as to offset this imbalance with the hope of arriving and a more robust and fair decision boundary than you would otherwise.

Resampling techniques are divided in two categories:
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

Example:
![SMOTE comparison](http://i.imgur.com/s8JHWPp.png)

This is a work in progress. Any comments, suggestions or corrections are welcome.

Dependencies:
* Numpy
* Scikit-Learn

References:

* NearMiss - "kNN approach to unbalanced data distributions: A case study involving information extraction" by Zhang et al.

* CNN - "Addressing the Curse of Imbalanced Training Sets: One-Sided Selection" by Kubat et al.

* One-Sided Selection - "Addressing the Curse of Imbalanced Training Sets: One-Sided Selection" by Kubat et al.

* NCL - "Improving identification of difficult small classes by balancing class distribution" by Laurikkala et al.

* SMOTE - "SMOTE: synthetic minority over-sampling technique" by Chawla, N.V et al.

* Borderline SMOTE -  "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning, Hui Han, Wen-Yuan Wang, Bing-Huan Mao"

* SVM_SMOTE - "Borderline Over-sampling for Imbalanced Data Classification, Nguyen, Cooper, Kamei"
