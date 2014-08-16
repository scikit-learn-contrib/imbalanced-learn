from __future__ import division
from __future__ import print_function

__author__ = 'fnogueira'

s = {
  "lines.linewidth": 2.0,
  "examples.download": True,
  "patch.linewidth": 0.5,
  "legend.fancybox": True,
  "axes.color_cycle": [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b"
  ],
  "axes.facecolor": "#f0f0f0",
  "axes.labelsize": "large",
  "axes.axisbelow": True,
  "axes.grid": True,
  "patch.edgecolor": "#f0f0f0",
  "axes.titlesize": "x-large",
  "svg.embed_char_paths": "path",
  "examples.directory": "",
  "figure.facecolor": "#f0f0f0",
  "grid.linestyle": "-",
  "grid.linewidth": 1.0,
  "grid.color": "#cbcbcb",
  "axes.edgecolor":"#f0f0f0",
  "xtick.major.size": 0,
  "xtick.minor.size": 0,
  "ytick.major.size": 0,
  "ytick.minor.size": 0,
  "axes.linewidth": 3.0,
  "font.size":14.0,
  "lines.linewidth": 4,
  "lines.solid_capstyle": "butt",
  "savefig.edgecolor": "#f0f0f0",
  "savefig.facecolor": "#f0f0f0",
  "figure.subplot.left"    : 0.08,
  "figure.subplot.right"   : 0.95,
  "figure.subplot.bottom"  : 0.07
}

def vizualization1():

    x, y = make_classification(n_classes=3, class_sep=3, weights=[0.2, 0.1, 0.7], \
                               n_informative=2, n_redundant=0, flip_y=0,\
                               n_features=2, n_clusters_per_class=1,\
                               n_samples=500, random_state=10)


    uc = UnbalancedDataset(random_state=2)
    uc.fit(x, y)

    ux, uy = uc.UnderSampler(ratio = 1)
    ox, oy = uc.OverSampler(ratio = 1)
    sx, sy = uc.smote(k= 5)
    bx, by = uc.b_smote1()
    cx, cy = uc.ClusterCentroids(ratio = 1, kargs={'n_jobs' : -1})
    svmx, svmy = uc.SVM_smote(svm_args={'class_weight' : 'auto'})


    # -------------------------------- // -------------------------------- #
    # Visualization

    f, ax = plt.subplots(2, 2, figsize=(14, 10))
    f.suptitle("Unbalanced Classification", fontsize=16)

    for e, c in zip(set(y), ['r', 'b', 'g']):
        ax[0, 0].scatter(x[y==e, 0], x[y==e, 1], color = c)
    ax[0, 0].set_title('Original', fontsize=12)


    for e, c in zip(set(y), ['r', 'b', 'g']):
        ax[0, 1].scatter(x[y==e, 0], x[y==e, 1], color = c)
    ax[0, 1].scatter(sx[:, 0], sx[:, 1], color = 'y', alpha = 0.5)
    ax[0, 1].set_title('SMOTE', fontsize=12)


    for e, c in zip(set(uy), ['r', 'b', 'g']):
        ax[1, 0].scatter(ux[uy==e, 0], ux[uy==e, 1], color = c)
    ax[1, 0].set_title('Under Sampling', fontsize=12)

    '''
    for e, c in zip(set(oy), ['r', 'b', 'g']):
        ax[1, 1].scatter(cx[cy==e, 0], cx[cy==e, 1], color = c)
    ax[1, 1].set_title('Cluster Centroids Under Sampling', fontsize=12)
    '''

    for e, c in zip(set(y), ['r', 'b', 'g']):
        ax[1, 1].scatter(x[y==e, 0], x[y==e, 1], color = c)
    ax[1, 1].scatter(svmx[:, 0], svmx[:, 1], color = 'y', alpha = 0.5)
    ax[1, 1].set_title('SMOTE', fontsize=12)

    # Hide ticks
    plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in ax[:, 1]], visible=False)

    plt.show()


def vizualization2():

    x, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],\
                               n_informative=3, n_redundant=1, flip_y=0,\
                               n_features=20, n_clusters_per_class=1,\
                               n_samples=5000, random_state=10)


    ud = UnbalancedDataset(random_state=2)
    ud.fit(x, y)

    ox, oy = ud.OverSampler()
    sx, sy = ud.smote(k = 5)
    bx1, by1 = ud.b_smote1(k = 5, m = 10)
    bx2, by2 = ud.b_smote2(k = 5, m = 10)
    svmx, svmy = ud.SVM_smote(svm_args={'class_weight' : 'auto'})


    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2)

    x = pca.fit_transform(x)

    ox = pca.transform(ox)
    sx = pca.transform(sx)
    bx1 = pca.transform(bx1)
    bx2 = pca.transform(bx2)
    svmx = pca.transform(svmx)

    # -------------------------------- // -------------------------------- #
    # Visualization

    f, ax = plt.subplots(2, 3, figsize=(16, 9))
    f.suptitle("Over-sampling with SMOTE: comparison", fontsize=16)

    for e, c in zip(set(y), ['purple', 'g']):
        ax[0, 0].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[0, 0].set_title('Original', fontsize=12)

    for e, c in zip(set(y), ['purple', 'g']):
        ax[0, 1].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[0, 1].scatter(ox[:, 0], ox[:, 1], color = 'y', alpha = 0.3)
    ax[0, 1].set_title('Random Over-sampling', fontsize=12)


    for e, c in zip(set(y), ['purple', 'g']):
        ax[0, 2].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[0, 2].scatter(sx[:, 0], sx[:, 1], color = 'y', alpha = 0.3)
    ax[0, 2].set_title('SMOTE', fontsize=12)


    for e, c in zip(set(y), ['purple', 'g']):
        ax[1, 0].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[1, 0].scatter(bx1[:, 0], bx1[:, 1], color = 'y', alpha = 0.3)
    ax[1, 0].set_title('Borderline-SMOTE type 1', fontsize=12)

    for e, c in zip(set(y), ['purple', 'g']):
        ax[1, 1].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[1, 1].scatter(bx2[:, 0], bx2[:, 1], color = 'y', alpha = 0.3)
    ax[1, 1].set_title('Borderline-SMOTE type 2', fontsize=12)

    for e, c in zip(set(y), ['purple', 'g']):
        ax[1, 2].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[1, 2].scatter(svmx[:, 0], svmx[:, 1], color = 'y', alpha = 0.3)
    ax[1, 2].set_title('SVM-SMOTE', fontsize=12)

    # Hide ticks
    plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in ax[:, 1]], visible=False)
    plt.setp([a.get_yticklabels() for a in ax[:, 2]], visible=False)


    plt.show()

if __name__ == '__main__':

    from UnbalancedClassification import UnbalancedDataset
    from sklearn.datasets import make_classification


    import matplotlib
    matplotlib.rcParams.update(s)

    import matplotlib.pyplot as plt

    vizualization2()

