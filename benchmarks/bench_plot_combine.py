"""Benchmarks of combination of over- and under-sampling methods.
"""
from __future__ import print_function

import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

lines_marker = ['-', '-.', '--', ':']

import numpy as np

from time import time

from imblearn import combine
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

STORE_PATH = './results/combine'
N_JOBS = -1

# Check that the storage path is existing
if not os.path.exists(STORE_PATH):
    os.makedirs(STORE_PATH)

# Fetch the dataset if not done already done
dataset = fetch_benchmark()

# Create the combine-sampling objects
combine_samplers = [combine.SMOTEENN(n_jobs=N_JOBS),
                    combine.SMOTETomek(n_jobs=N_JOBS)]
combine_samplers_legend = ['SMOTE+ENN', 'SMOTE+TL']

# Set the number of color in the palette depending of the number of methods
sns.palplot(sns.color_palette("hls", len(combine_samplers_legend)))

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
    for comb in combine_samplers:
        pipelines.append(make_pipeline(comb, cl))

datasets_nb_samples = []
datasets_time = []
# For each dataset
for idx_dataset, current_set in enumerate(dataset):

    print('Process the dataset {}/{}'.format(idx_dataset+1, len(dataset)))

    # Apply sttratified k-fold cross-validation
    skf = StratifiedKFold(current_set['label'])

    # For each pipeline, make the classification
    pipeline_tpr_mean = []
    pipeline_tpr_std = []
    pipeline_auc = []
    pipeline_auc_std = []
    pipeline_time = []
    pipeline_nb_samples = []
    for idx_pipe, pipe in enumerate(pipelines):
        print('Pipeline {}/{}'.format(idx_pipe+1, len(pipelines)))
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

            # Launch the time to check the performance of each combine-sampler
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

    # For each classifier make a different plot
    for cl_idx in range(len(classifiers)):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # For each combine-sampling methods
        for comb_idx in range(len(combine_samplers)):

            # Find the linear index for the pipeline
            idx_pipeline = cl_idx * len(combine_samplers) + comb_idx

            ax.plot(mean_fpr, pipeline_tpr_mean[idx_pipeline],
                    label=(combine_samplers_legend[comb_idx] +
                           r' - AUC $= {:1.3f} \pm {:1.3f}$'.format(
                               pipeline_auc[idx_pipeline],
                               pipeline_auc_std[idx_pipeline])),
                    lw=2, linestyle=lines_marker[comb_idx%len(lines_marker)])

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

    # Keep only the interesting data
    datasets_nb_samples.append(np.mean(pipeline_nb_samples))
    datasets_time.append(pipeline_time[:len(combine_samplers)])

datasets_time = np.array(datasets_time)
datasets_nb_samples = np.array(datasets_nb_samples)

fig = plt.figure()
ax = fig.add_subplot(111)

for comb_idx in range(len(combine_samplers)):
    ax.plot(datasets_nb_samples, datasets_time[:, comb_idx],
            label=combine_samplers_legend[comb_idx],
            lw=2, linestyle=lines_marker[comb_idx%len(lines_marker)])
    plt.xlabel('# samples')
    plt.ylabel('Time (s)')
    plt.title('Complexity time of the different combining over- and'
              ' under-sampling methods')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='lower right')

# Save the plot
plt.savefig(os.path.join(STORE_PATH, 'complexity.pdf'),
            bbox_extra_artists=(lgd,),
            bbox_inches='tight')
