"""Benchmarks of the under-sampling methods.
"""

import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from time import time

from imblearn import under_sampling
from imblearn.datasets import fetch_benchmark
from imblearn.pipeline import make_pipeline

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from scipy import interp

STORE_PATH = './results/under-sampling'
N_JOBS = -1

# Check that the storage path is existing
if not os.path.exists(STORE_PATH):
    os.makedirs(STORE_PATH)

# Fetch the dataset if not done already done
dataset = fetch_benchmark()

# Create the under-sampling objects
under_samplers = [under_sampling.ClusterCentroids(n_jobs=N_JOBS),
                  under_sampling.CondensedNearestNeighbour(n_jobs=N_JOBS),
                  under_sampling.EditedNearestNeighbours(n_jobs=N_JOBS),
                  under_sampling.RepeatedEditedNearestNeighbours(
                      n_jobs=N_JOBS),
                  under_sampling.AllKNN(n_jobs=N_JOBS),
                  under_sampling.InstanceHardnessThreshold(n_jobs=N_JOBS),
                  under_sampling.NearMiss(version=1, n_jobs=N_JOBS),
                  under_sampling.NearMiss(version=2, n_jobs=N_JOBS),
                  under_sampling.NearMiss(version=3, n_jobs=N_JOBS),
                  under_sampling.NeighbourhoodCleaningRule(n_jobs=N_JOBS),
                  under_sampling.OneSidedSelection(n_jobs=N_JOBS),
                  under_sampling.RandomUnderSampler(),
                  under_sampling.TomekLinks(n_jobs=N_JOBS)]
under_samplers_legend = ['CC', 'CNN', 'ENN', 'RENN', 'AkNN', 'IHT', 'NM1',
                         'NM2', 'NM3', 'NCR', 'OSS', 'RUS', 'TL']

# Create the classifier objects
classifiers = [RandomForestClassifier(n_jobs=N_JOBS),
               GradientBoostingClassifier(),
               GaussianNB(),
               KNeighborsClassifier(n_jobs=N_JOBS),
               DecisionTreeClassifier()]
classifiers_legend = ['RF', 'GB', 'NB', 'kNN', 'DT']

# Create the diffrent pipeline
pipelines = []
for cl in classifiers:
    for us in under_samplers:
        pipelines.append(make_pipeline(us, cl))

datasets_nb_samples = []
datasets_time = []
# For each dataset
for idx_dataset, current_set in enumerate(dataset):

    # Apply sttratified k-fold cross-validation
    skf = StratifiedKFold(current_set['label'])

    # For each pipeline, make the classification
    pipeline_tpr_mean = []
    pipeline_tpr_std = []
    pipeline_auc = []
    pipeline_auc_std = []
    pipeline_time = []
    pipeline_nb_samples = []
    for pipe in pipelines:
        # For each fold from the cross-validation
        mean_tpr = []
        mean_fpr = np.linspace(0, 1, 30)
        cv_auc = []
        cv_time = []
        cv_nb_samples = []
        for train_index, test_index in skf:
            # Extract the data
            X_train, X_test = (current_set['data'][train_index],
                               current_set['data'][test_index])
            y_train, y_test = (current_set['label'][train_index],
                               current_set['label'][test_index])
            cv_nb_samples.append(y_train.size)

            # Launch the time to check the performance of each under-sampler
            tstart = time()
            # Fit the pipeline on the training_set
            pipe.fit(X_train, y_train)
            # Stop the timer
            elapsed_time = time() - tstart
            cv_time.append(elapsed_time)
            # Predict on the testing set
            y_hat = pipe.predict_proba(X_test)

            # Compute the statistics
            fpr, tpr, thresholds = roc_curve(y_test, y_hat[:, 1])
            mean_tpr.append(interp(mean_fpr, fpr, tpr))
            mean_tpr[-1][0] = 0.0
            cv_auc.append(auc(mean_fpr, mean_tpr[-1]))

        avg_tpr = np.mean(mean_tpr, axis=0)
        std_tpr = np.std(mean_tpr, axis=0)
        avg_tpr[-1] = 1.0
        pipeline_tpr_mean.append(avg_tpr)
        pipeline_tpr_std.append(std_tpr)
        pipeline_auc.append(auc(mean_fpr, avg_tpr))
        pipeline_auc_std.append(np.std(cv_auc))
        pipeline_time.append(np.mean(cv_time))
        pipeline_nb_samples.append(np.mean(cv_nb_samples))

    # Keep only the interesting data
    datasets_nb_samples.append(np.mean(pipeline_nb_samples))
    datasets_time.append(pipeline_time[:len(under_samplers)])

    # For each classifier make a different plot
    for cl_idx in range(len(classifiers)):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # For each under-sampling methods
        for us_idx in range(len(under_samplers)):

            # Find the linear index for the pipeline
            idx_pipeline = cl_idx * len(under_samplers) + us_idx

            ax.plot(mean_fpr, pipeline_tpr_mean[idx_pipeline],
                    label=(under_samplers_legend[us_idx] +
                           r' - AUC $= {:1.3f} \pm {:1.3f}$'.format(
                               pipeline_auc[idx_pipeline],
                               pipeline_auc_std[idx_pipeline])),
                    lw=2)
#             ax.fill_between(mean_fpr,
#                             (pipeline_tpr_mean[idx_pipeline] +
#                              pipeline_tpr_std[idx_pipeline]),
#                             (pipeline_tpr_mean[idx_pipeline] -
#                              pipeline_tpr_std[idx_pipeline]),
#                             alpha=0.2)


        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC analysis using a ' + classifiers_legend[cl_idx] +
                  ' classifier')
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='lower right')

        # Save the plot
        plt.savefig(os.path.join(STORE_PATH, 'x{}data_{}.pdf'.format(
            idx_dataset,
            classifiers_legend[cl_idx])),
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight')

datasets_time = np.array(datasets_time)
datasets_nb_samples = np.array(datasets_nb_samples)

fig = plt.figure()
ax = fig.add_subplot(111)

for us_idx in range(len(under_samplers)):
    ax.plot(datasets_nb_samples[:, us_idx], datasets_time,
             label=under_samplers_legend[us_idx])
    plt.xlabel('# samples')
    plt.ylabel('Time (s)')
    plt.title('Complexity time of the different under-sampling methods')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='lower right')

# Save the plot
plt.savefig(os.path.join(STORE_PATH, 'complexity.pdf'),
            bbox_extra_artists=(lgd,),
            bbox_inches='tight')
