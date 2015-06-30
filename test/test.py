from __future__ import print_function
__author__ = 'fnogueira, glemaitre'

from sklearn.datasets import make_classification

from unbalanced_dataset.unbalanced_dataset import UnbalancedDataset

from unbalanced_dataset.over_sampling import OverSampler
from unbalanced_dataset.over_sampling import SMOTE

from unbalanced_dataset.under_sampling import UnderSampler
from unbalanced_dataset.under_sampling import TomekLinks
from unbalanced_dataset.under_sampling import ClusterCentroids
from unbalanced_dataset.under_sampling import NearMiss
from unbalanced_dataset.under_sampling import CondensedNearestNeighbour
from unbalanced_dataset.under_sampling import OneSidedSelection
from unbalanced_dataset.under_sampling import NeighbourhoodCleaningRule

from unbalanced_dataset.ensemble_sampling import EasyEnsemble
from unbalanced_dataset.ensemble_sampling import BalanceCascade

from unbalanced_dataset.pipeline import SMOTEENN
from unbalanced_dataset.pipeline import SMOTETomek

# Generate some data
print('Generate samples using scikit-learn')
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=10)

verbose = True

def test_smote(x, y):
    print('SMOTE')
    sm = SMOTE(kind='regular', verbose=verbose)
    svmx, svmy = sm.fit_transform(x, y)

    print('SMOTE bordeline 1')
    sm = SMOTE(kind='borderline1', verbose=verbose)
    svmx, svmy = sm.fit_transform(x, y)

    print('SMOTE bordeline 2')
    sm = SMOTE(kind='borderline2', verbose=verbose)
    svmx, svmy = sm.fit_transform(x, y)

    print('SMOTE SVM')
    svm_args={'class_weight': 'auto'}
    sm = SMOTE(kind='svm', verbose=verbose, **svm_args)
    svmx, svmy = sm.fit_transform(x, y)


def test_rest(x, y):

    print('Random under-sampling')
    US = UnderSampler(verbose=verbose)
    usx, usy = US.fit_transform(x, y)

    print('Tomek links')
    TL = TomekLinks(verbose=verbose)
    tlx, tly = TL.fit_transform(x, y)

    print('Clustering centroids')
    CC = ClusterCentroids(verbose=verbose)
    ccx, ccy = CC.fit_transform(x, y)

    print('NearMiss-1')
    NM1 = NearMiss(version=1, verbose=verbose)
    nm1x, nm1y = NM1.fit_transform(x, y)

    print('NearMiss-2')
    NM2 = NearMiss(version=2, verbose=verbose)
    nm2x, nm2y = NM2.fit_transform(x, y)

    print('NearMiss-3')
    NM3 = NearMiss(version=3, verbose=verbose)
    nm3x, nm3y = NM3.fit_transform(x, y)

    print('Neighboorhood Cleaning Rule')
    NCR = NeighbourhoodCleaningRule(verbose=verbose)
    ncrx, ncry = NCR.fit_transform(x, y)

    print('Random over-sampling')
    OS = OverSampler(verbose=verbose)
    ox, oy = OS.fit_transform(x, y)

    print('SMOTE Tomek links')
    STK = SMOTETomek(verbose=verbose)
    stkx, stky = STK.fit_transform(x, y)

    print('SMOTE ENN')
    SENN = SMOTEENN(verbose=verbose)
    sennx, senny = SENN.fit_transform(x, y)

    print('EasyEnsemble')
    EE = EasyEnsemble(verbose=verbose)
    eex, eey = EE.fit_transform(x, y)


def test_CNN(x, y):
    print('Condensed Nearest Neighbour')
    CNN = CondensedNearestNeighbour(verbose=verbose)
    cnnx, cnny = CNN.fit_transform(x, y)

    print('One-Sided Selection')
    OSS = OneSidedSelection(verbose=verbose)
    ossx, ossy = OSS.fit_transform(x, y)

    print('BalanceCascade')
    BS = BalanceCascade(verbose=verbose)
    bsx, bsy = BS.fit_transform(x, y)


if __name__ == '__main__':

    test_smote(X, Y)
    test_rest(X, Y)
    test_CNN(X, Y)
