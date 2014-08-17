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

* Over-sampling
    1. Random minority over-sampling with replacement
    2. SMOTE - Synthetic Minority Over-sampling Technique
    3. bSMOTE(1&2) - Borderline SMOTE of types 1 and 2
    4. SVM_SMOTE - Support Vectors SMOTE

This is a work in progress. Any comments, suggestions or corrections are welcome.

Dependenceies:
* Numpy
* Scikit-Learn

References:

* SMOTE - "SMOTE: synthetic minority over-sampling technique" by Chawla, N.V et al.

* Borderline SMOTE -  "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning, Hui Han, Wen-Yuan Wang, Bing-Huan Mao"

* SVM_SMOTE - "Borderline Over-sampling for Imbalanced Data Classification, Nguyen, Cooper, Kamei"
