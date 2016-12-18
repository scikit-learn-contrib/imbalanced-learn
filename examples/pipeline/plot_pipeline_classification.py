"""

=========================
Pipeline Object
=========================

An example of the Pipeline object working with transformers and resamplers.

"""

from sklearn.cross_validation import train_test_split as tts
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier as KNN

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours)

print(__doc__)

# Generate the dataset
X, y = make_classification(n_classes=2, class_sep=1.25, weights=[0.3, 0.7],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=5, n_clusters_per_class=1,
                           n_samples=5000, random_state=10)

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)

# Create the samplers
enn = EditedNearestNeighbours()
renn = RepeatedEditedNearestNeighbours()

# Create teh classifier
knn = KNN(1)


# Make the splits
X_train, X_test, y_train, y_test = tts(X, y, random_state=42)

# Add one transformers and two samplers in the pipeline object
pipeline = make_pipeline(pca, enn, renn, knn)

pipeline.fit(X_train, y_train)
y_hat = pipeline.predict(X_test)

print(classification_report(y_test, y_hat))
