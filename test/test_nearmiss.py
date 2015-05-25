import matplotlib.pyplot as plt

from UnbalancedDataset import NearMiss, CondensedNearestNeighbour
from sklearn.datasets import make_classification

# Generate some data
x, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],\
                           n_informative=3, n_redundant=1, flip_y=0,\
                           n_features=20, n_clusters_per_class=1,\
                           n_samples=5000, random_state=10)

# # Try NearMiss algorithm
# NM1 = NearMiss(random_state=1, version=1, metric='l1')
# nm1x, nm1y = NM1.fit_transform(x, y)

# NM2 = NearMiss(random_state=1, version=2)
# nm2x, nm2y = NM2.fit_transform(x, y)

# NM3 = NearMiss(random_state=1, version=3)
# nm3x, nm3y = NM3.fit_transform(x, y)

# Try CNN
# CNN = CondensedNearestNeighbour(random_state=1, n_seeds_S=20)
# cnnx, cnny = CNN.fit_transform(x, y)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x = pca.fit_transform(x)
# nm1x = pca.transform(nm1x)
# nm2x = pca.transform(nm2x)
# nm3x = pca.transform(nm3x)
# cnnx = pca.transform(cnnx)

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
