"""
====================================
Usage of pipeline embedding samplers
====================================

An example of the :class:~imblearn.pipeline.Pipeline` object (or
:func:`~imblearn.pipeline.make_pipeline` helper function) working with
transformers and resamplers.
"""

# Authors: Christos Aridas
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

# %%
print(__doc__)

# %% [markdown]
# Let's first create an imbalanced dataset and split in to two sets.

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_classes=2,
    class_sep=1.25,
    weights=[0.3, 0.7],
    n_informative=3,
    n_redundant=1,
    flip_y=0,
    n_features=5,
    n_clusters_per_class=1,
    n_samples=5000,
    random_state=10,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# %% [markdown]
# Now, we will create each individual steps that we would like later to combine

# %%
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

pca = PCA(n_components=2)
enn = EditedNearestNeighbours()
smote = SMOTE(random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)

# %% [markdown]
# Now, we can finally create a pipeline to specify in which order the different
# transformers and samplers should be executed before to provide the data to
# the final classifier.

# %%
from imblearn.pipeline import make_pipeline

model = make_pipeline(pca, enn, smote, knn)

# %% [markdown]
# We can now use the pipeline created as a normal classifier where resampling
# will happen when calling `fit` and disabled when calling `decision_function`,
# `predict_proba`, or `predict`.

# %%
from sklearn.metrics import classification_report

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
