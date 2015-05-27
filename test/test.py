__author__ = 'fnogueira, glemaitre'

from sklearn.datasets import make_classification

from UnbalancedDataset import UnderSampler, NearMiss, CondensedNearestNeighbour, OneSidedSelection, NeighboorhoodCleaningRule, TomekLinks, ClusterCentroids, OverSampler, SMOTE, bSMOTE1, bSMOTE2, SVM_SMOTE, SMOTETomek, SMOTEENN, EasyEnsemble, BalanceCascade


# Generate some data
print 'Generate samples using scikit-learn'
x, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],\
                           n_informative=3, n_redundant=1, flip_y=0,\
                           n_features=20, n_clusters_per_class=1,\
                           n_samples=5000, random_state=10)

verbose = False

print 'Random under-sampling'
US = UnderSampler(verbose=verbose)
usx, usy = US.fit_transform(x, y)

print 'Tomek links'
TL = TomekLinks(verbose=verbose)
tlx, tly = TL.fit_transform(x, y)

print 'Clustering centroids'
CC = ClusterCentroids(verbose=verbose)
ccx, ccy = CC.fit_transform(x, y)

print 'NearMiss-1'
NM1 = NearMiss(version=1, verbose=verbose)
nm1x, nm1y = NM1.fit_transform(x, y)

print 'NearMiss-2'
NM2 = NearMiss(version=2, verbose=verbose)
nm2x, nm2y = NM2.fit_transform(x, y)

print 'NearMiss-3'
NM3 = NearMiss(version=3, verbose=verbose)
nm3x, nm3y = NM3.fit_transform(x, y)

print 'Condensed Nearest Neighbour'
CNN = CondensedNearestNeighbour(verbose=verbose)
cnnx, cnny = CNN.fit_transform(x, y)

print 'One-Sided Selection'
OSS = OneSidedSelection(verbose=verbose)
ossx, ossy = OSS.fit_transform(x, y)

print 'Neighboorhood Cleaning Rule'
NCR = NeighboorhoodCleaningRule(verbose=verbose)
ncrx, ncry = NCR.fit_transform(x, y) 

print 'Random over-sampling'
OS = OverSampler(verbose=verbose)
ox, oy = OS.fit_transform(x, y)

print 'SMOTE'
smote = SMOTE(verbose=verbose)
sx, sy = smote.fit_transform(x, y)

print 'SMOTE bordeline 1'
bsmote1 = bSMOTE1(verbose=verbose)
bsx1, bsy1 = bsmote1.fit_transform(x, y)

print 'SMOTE bordeline 2'
bsmote2 = bSMOTE2(verbose=verbose)
bsx2, bsy2 = bsmote2.fit_transform(x, y)

print 'SMOTE SVM'
svm_args={'class_weight' : 'auto'}
svmsmote = SVM_SMOTE(random_state=1, verbose=verbose, **svm_args)

print 'SMOTE Tomek links'
STK = SMOTETomek(verbose=verbose)
stkx, stky = STK.fit_transform(x, y)

print 'SMOTE ENN'
SENN = SMOTEENN(verbose=verbose)
sennx, senny = SENN.fit_transform(x, y)

print 'EasyEnsemble'
EE = EasyEnsemble(verbose=verbose)
eex, eey = EE.fit_transform(x, y)

print 'BalanceCascade'
BS = BalanceCascade(verbose=verbose)
bsx, bsy = BS.fit_transform(x, y)
