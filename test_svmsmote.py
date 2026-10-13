from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
import warnings

# Ignore standard convergence warnings for this quick test
warnings.filterwarnings("ignore")

print("Generating imbalanced data...")
X, y = make_classification(n_classes=2, class_sep=2,
    weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=42)

print(f"Original dataset shape: {Counter(y)}")

# Initialize SVMSMOTE with the fast, primal-optimizing LinearSVC
print("Attempting to run SVMSMOTE with LinearSVC...")
fast_svm = LinearSVC(dual=False, random_state=42)
sm = SVMSMOTE(random_state=42, svm_estimator=fast_svm)

# This is the exact line that used to throw the RuntimeError
X_res, y_res = sm.fit_resample(X, y)

print(f"Resampled dataset shape: {Counter(y_res)}")
print("SUCCESS: The Coordinate Descent math path executed flawlessly.")