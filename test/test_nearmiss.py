import matplotlib.pyplot as plt

from UnbalancedDataset import NearMiss
from sklearn.datasets import make_classification

# Generate some data
x, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],\
                           n_informative=3, n_redundant=1, flip_y=0,\
                           n_features=20, n_clusters_per_class=1,\
                           n_samples=5000, random_state=10)

# Try NearMiss algorithm
NM = NearMiss(random_state=1)
ox, oy = NM.fit_transform(x, y)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x = pca.fit_transform(x)
ox = pca.transform(ox)

plt.figure()
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1], color='r')
plt.show()


plt.figure()
plt.scatter(ox[oy==0, 0], ox[oy==0, 1])
plt.scatter(ox[oy==1, 0], ox[oy==1, 1], color='r')
plt.show()
