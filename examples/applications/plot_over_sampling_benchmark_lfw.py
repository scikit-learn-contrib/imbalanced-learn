"""
==========================================================
Benchmark over-sampling methods in a face recognition task
==========================================================

In this face recognition example two faces are used from the LFW
(Faces in the Wild) dataset. Several implemented over-sampling
methods are used in conjunction with a 3NN classifier in order
to examine the improvement of the classifier's output quality
by using an over-sampler.
"""

# Authors: Christos Aridas
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

# %%
print(__doc__)

import seaborn as sns

sns.set_context("poster")

# %% [markdown]
# Load the dataset
# ----------------
#
# We will use a dataset containing image from know person where we will
# build a model to recognize the person on the image. We will make this problem
# a binary problem by taking picture of only George W. Bush and Bill Clinton.

# %%
import numpy as np
from sklearn.datasets import fetch_lfw_people

data = fetch_lfw_people()
george_bush_id = 1871  # Photos of George W. Bush
bill_clinton_id = 531  # Photos of Bill Clinton
classes = [george_bush_id, bill_clinton_id]
classes_name = np.array(["B. Clinton", "G.W. Bush"], dtype=object)

# %%
mask_photos = np.isin(data.target, classes)
X, y = data.data[mask_photos], data.target[mask_photos]
y = (y == george_bush_id).astype(np.int8)
y = classes_name[y]

# %% [markdown]
# We can check the ratio between the two classes.

# %%
import matplotlib.pyplot as plt
import pandas as pd

class_distribution = pd.Series(y).value_counts(normalize=True)
ax = class_distribution.plot.barh()
ax.set_title("Class distribution")
pos_label = class_distribution.idxmin()
plt.tight_layout()
print(f"The positive label considered as the minority class is {pos_label}")

# %% [markdown]
# We see that we have an imbalanced classification problem with ~95% of the
# data belonging to the class G.W. Bush.
#
# Compare over-sampling approaches
# --------------------------------
#
# We will use different over-sampling approaches and use a kNN classifier
# to check if we can recognize the 2 presidents. The evaluation will be
# performed through cross-validation and we will plot the mean ROC curve
# using `skore.evaluate`.
#
# We will create different pipelines and evaluate them.

# %%
import skore
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from imblearn import FunctionSampler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline

classifier = KNeighborsClassifier(n_neighbors=3)

pipelines = {
    "No resampling": make_pipeline(FunctionSampler(), classifier),
    "Random Over-Sampler": make_pipeline(
        RandomOverSampler(random_state=42), classifier
    ),
    "ADASYN": make_pipeline(ADASYN(random_state=42), classifier),
    "SMOTE": make_pipeline(SMOTE(random_state=42), classifier),
}

# %% [markdown]
# We use `skore.evaluate` to evaluate each pipeline using a
# :class:`~sklearn.model_selection.StratifiedKFold` cross-validation and
# compare their performance.

# %%
cv = StratifiedKFold(n_splits=3)

reports = {}
for name, model in pipelines.items():
    reports[name] = skore.evaluate(model, X, y, splitter=cv, pos_label=pos_label)

# %%
import pandas as pd

results = {name: r.metrics.summarize().frame() for name, r in reports.items()}
pd.concat(results)

# %% [markdown]
# We can also plot the ROC curves for each pipeline.

# %%
fig, ax = plt.subplots(figsize=(9, 9))
for name, report in reports.items():
    report.metrics.roc().plot()
plt.show()

# %% [markdown]
# We see that for this task, methods that are generating new samples with some
# interpolation (i.e. ADASYN and SMOTE) perform better than random
# over-sampling or no resampling.
