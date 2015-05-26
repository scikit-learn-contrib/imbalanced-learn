import matplotlib.pyplot as plt

import numpy as np

from UnbalancedDataset import NearMiss, CondensedNearestNeighbour, OneSidedSelection, NeighboorhoodCleaningRule, SMOTE, SMOTETomek, SMOTEENN, UnderSampler, EasyEnsemble, BalanceCascade
from sklearn.datasets import make_classification

# Generate some data
x, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],\
                           n_informative=3, n_redundant=1, flip_y=0,\
                           n_features=20, n_clusters_per_class=1,\
                           n_samples=5000, random_state=10)

r = float(np.count_nonzero(y == 1)) / float(np.count_nonzero(y == 0))

# Try NearMiss algorithm
NM1 = NearMiss(random_state=1, version=1, metric='l1')
nm1x, nm1y = NM1.fit_transform(x, y)

NM2 = NearMiss(random_state=1, version=2)
nm2x, nm2y = NM2.fit_transform(x, y)

NM3 = NearMiss(random_state=1, version=3)
nm3x, nm3y = NM3.fit_transform(x, y)

# Try CNN
CNN = CondensedNearestNeighbour(random_state=1, n_seeds_S=20)
cnnx, cnny = CNN.fit_transform(x, y)

# Try OSS
OSS = OneSidedSelection(random_state=1, n_seeds_S=20)
ossx, ossy = OSS.fit_transform(x, y)

# Try NCR
NCR = NeighboorhoodCleaningRule(random_state=1, size_ngh=51)
ncrx, ncry = NCR.fit_transform(x, y) 

# Try SMOTE
smote = SMOTE(random_state=1)
sx, sy = smote.fit_transform(x, y)

# Try SMOTE Tomek
STK = SMOTETomek(random_state=1)
stkx, stky = STK.fit_transform(x, y)

# Try SMOTE ENN
SENN = SMOTEENN(random_state=1, ratio=r, size_ngh=51)
sennx, senny = SENN.fit_transform(x, y)

# Try Undersampling
USS = UnderSampler(random_state=1, replacement=False)
ussx, ussy = USS.fit_transform(x, y)

# Try EasyEnsemble
EE = EasyEnsemble(random_state=1)
eex, eey = EE.fit_transform(x, y)

# Try BalanceCascade
classifier_opts = {'n_jobs': 4}
BS = BalanceCascade(random_state=1, classifier='adaboost', bootstrap=True)
bsx, bsy = BS.fit_transform(x, y)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x = pca.fit_transform(x)
# nm1x = pca.transform(nm1x)
# nm2x = pca.transform(nm2x)
# nm3x = pca.transform(nm3x)
# cnnx = pca.transform(cnnx)
# ossx = pca.transform(ossx)
# ncrx = pca.transform(ncrx)
# sx = pca.transform(sx)
# stkx = pca.transform(stkx)
# sennx = pca.transform(sennx)

# plt.figure()
# plt.scatter(nm1x[nm1y==0, 0], nm1x[nm1y==0, 1])
# plt.scatter(nm1x[nm1y==1, 0], nm1x[nm1y==1, 1], color='r')
# plt.show()

# plt.figure()
# plt.scatter(nm2x[nm2y==0, 0], nm2x[nm2y==0, 1])
# plt.scatter(nm2x[nm2y==1, 0], nm2x[nm2y==1, 1], color='r')
# plt.show()

# plt.figure()
# plt.scatter(nm3x[nm3y==0, 0], nm3x[nm3y==0, 1])
# plt.scatter(nm3x[nm3y==1, 0], nm3x[nm3y==1, 1], color='r')
# plt.show()

# plt.figure()
# plt.scatter(cnnx[cnny==0, 0], cnnx[cnny==0, 1])
# plt.scatter(cnnx[cnny==1, 0], cnnx[cnny==1, 1], color='r')
# plt.show()

# plt.figure()
# plt.scatter(ossx[ossy==0, 0], ossx[ossy==0, 1])
# plt.scatter(ossx[ossy==1, 0], ossx[ossy==1, 1], color='r')
# plt.show()

# plt.figure()
# plt.scatter(ncrx[ncry==0, 0], ncrx[ncry==0, 1])
# plt.scatter(ncrx[ncry==1, 0], ncrx[ncry==1, 1], color='r')
# plt.show()

# plt.figure()
# plt.scatter(sx[sy==0, 0], sx[sy==0, 1])
# plt.scatter(sx[sy==1, 0], sx[sy==1, 1], color='r')
# plt.show()

# plt.figure()
# plt.scatter(stkx[stky==0, 0], stkx[stky==0, 1])
# plt.scatter(stkx[stky==1, 0], stkx[stky==1, 1], color='r')
# plt.show()

# plt.figure()
# plt.scatter(sennx[senny==0, 0], sennx[senny==0, 1])
# plt.scatter(sennx[senny==1, 0], sennx[senny==1, 1], color='r')
# plt.show()
