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

def vizualization():

    from sklearn.datasets import make_classification
    x, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],\
                               n_informative=3, n_redundant=1, flip_y=0,\
                               n_features=20, n_clusters_per_class=1,\
                               n_samples=5000, random_state=10)



    from UnbalancedDataset import OverSampler, SMOTE, bSMOTE1, bSMOTE2, SVM_SMOTE

    ratio = 1
    new = ratio * 500

    # -------------------------------- // -------------------------------- #
    # Datasets
    OS = OverSampler(random_state=1)
    ox, oy = OS.fit_transform(x, y)

    smote = SMOTE(random_state=1)
    sx, sy = smote.fit_transform(x, y)

    bsmote1 = bSMOTE1(random_state=1)
    bsx1, bsy1 = bsmote1.fit_transform(x, y)

    bsmote2 = bSMOTE2(random_state=1)
    bsx2, bsy2 = bsmote2.fit_transform(x, y)

    svmsmote = SVM_SMOTE(random_state=1, svm_args={'class_weight' : 'auto'})
    svmx, svmy = svmsmote.fit_transform(x, y)


    # -------------------------------- // -------------------------------- #
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2)

    x = pca.fit_transform(x)

    ox = pca.transform(ox)
    sx = pca.transform(sx)
    bsx1 = pca.transform(bsx1)
    bsx2 = pca.transform(bsx2)
    svmx = pca.transform(svmx)

    # -------------------------------- // -------------------------------- #
    # Visualization
    import matplotlib
    matplotlib.rcParams.update(s)
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(2, 3, figsize=(16, 9))
    f.suptitle("Over-sampling with SMOTE: comparison", fontsize=16)

    for e, c in zip(set(y), ['purple', 'g']):
        ax[0, 0].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[0, 0].set_title('Original', fontsize=12)

    for e, c in zip(set(y), ['purple', 'g']):
        ax[0, 1].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[0, 1].scatter(ox[-new:, 0], ox[-new:, 1], color = 'y', alpha = 0.3)
    ax[0, 1].set_title('Random Over-sampling', fontsize=12)


    for e, c in zip(set(y), ['purple', 'g']):
        ax[0, 2].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[0, 2].scatter(sx[-new:, 0], sx[-new:, 1], color = 'y', alpha = 0.3)
    ax[0, 2].set_title('SMOTE', fontsize=12)


    for e, c in zip(set(y), ['purple', 'g']):
        ax[1, 0].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[1, 0].scatter(bsx1[-new:, 0], bsx1[-new:, 1], color = 'y', alpha = 0.3)
    ax[1, 0].set_title('Borderline-SMOTE type 1', fontsize=12)

    for e, c in zip(set(y), ['purple', 'g']):
        ax[1, 1].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[1, 1].scatter(bsx2[-new:, 0], bsx2[-new:, 1], color = 'y', alpha = 0.3)
    ax[1, 1].set_title('Borderline-SMOTE type 2', fontsize=12)

    for e, c in zip(set(y), ['purple', 'g']):
        ax[1, 2].scatter(x[y==e, 0], x[y==e, 1], color = c, alpha = 0.5)
    ax[1, 2].scatter(svmx[-new:, 0], svmx[-new:, 1], color = 'y', alpha = 0.3)
    ax[1, 2].set_title('SVM-SMOTE', fontsize=12)

    # Hide ticks
    plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in ax[:, 1]], visible=False)
    plt.setp([a.get_yticklabels() for a in ax[:, 2]], visible=False)

    plt.show()

if __name__ == '__main__':

    vizualization()

