"""
==========================================================
Porto Seguro: balancing samples in mini-batches with Keras
==========================================================

This example compares two strategies to train a neural-network on the Porto
Seguro Kaggle data set [1]_. The data set is imbalanced and we show that
balancing each mini-batch allows to improve performance and reduce the training
time.

References
----------

.. [1] https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

print(__doc__)

###############################################################################
# Data loading
###############################################################################

from collections import Counter
import pandas as pd
import numpy as np

###############################################################################
# First, you should download the Porto Seguro data set from Kaggle. See the
# link in the introduction.

training_data = pd.read_csv('./input/train.csv')
testing_data = pd.read_csv('./input/test.csv')

y_train = training_data[['id', 'target']].set_index('id')
X_train = training_data.drop(['target'], axis=1).set_index('id')
X_test = testing_data.set_index('id')

###############################################################################
# The data set is imbalanced and it will have an effect on the fitting.

print('The data set is imbalanced: {}'.format(Counter(y_train['target'])))

###############################################################################
# Define the pre-processing pipeline
###############################################################################

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer


def convert_float64(X):
    return X.astype(np.float64)


###############################################################################
# We want to standard scale the numerical features while we want to one-hot
# encode the categorical features. In this regard, we make use of the
# :class:`sklearn.compose.ColumnTransformer`.

numerical_columns = [name for name in X_train.columns
                     if '_calc_' in name and '_bin' not in name]
numerical_pipeline = make_pipeline(
    FunctionTransformer(func=convert_float64, validate=False),
    StandardScaler())

categorical_columns = [name for name in X_train.columns
                       if '_cat' in name]
categorical_pipeline = make_pipeline(
    SimpleImputer(missing_values=-1, strategy='most_frequent'),
    OneHotEncoder(categories='auto'))

preprocessor = ColumnTransformer(
    [('numerical_preprocessing', numerical_pipeline, numerical_columns),
     ('categorical_preprocessing', categorical_pipeline, categorical_columns)],
    remainder='drop')

# Create an environment variable to avoid using the GPU. This can be changed.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

###############################################################################
# Create a neural-network
###############################################################################

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization


def make_model(n_features):
    model = Sequential()
    model.add(Dense(200, input_shape=(n_features,),
              kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(50, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(25, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


###############################################################################
# We create a decorator to report the computation time

import time
from functools import wraps


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start_time = time.time()
        result = f(*args, **kwds)
        elapsed_time = time.time() - start_time
        print('Elapsed computation time: {:.3f} secs'
              .format(elapsed_time))
        return (elapsed_time, result)
    return wrapper


###############################################################################
# The first model will be trained using the ``fit`` method and with imbalanced
# mini-batches.

from sklearn.metrics import roc_auc_score


@timeit
def fit_predict_imbalanced_model(X_train, y_train, X_test, y_test):
    model = make_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=2, verbose=1, batch_size=1000)
    y_pred = model.predict_proba(X_test, batch_size=1000)
    return roc_auc_score(y_test, y_pred)


###############################################################################
# In the contrary, we will use imbalanced-learn to create a generator of
# mini-batches which will yield balanced mini-batches.

from imblearn.keras import BalancedBatchGenerator


@timeit
def fit_predict_balanced_model(X_train, y_train, X_test, y_test):
    model = make_model(X_train.shape[1])
    training_generator = BalancedBatchGenerator(X_train, y_train,
                                                batch_size=1000,
                                                random_state=42)
    model.fit_generator(generator=training_generator, epochs=5, verbose=1)
    y_pred = model.predict_proba(X_test, batch_size=1000)
    return roc_auc_score(y_test, y_pred)


###############################################################################
# Classification loop
###############################################################################

###############################################################################
# We will perform a 10-fold cross-validation and train the neural-network with
# the two different strategies previously presented.

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10)

cv_results_imbalanced = []
cv_time_imbalanced = []
cv_results_balanced = []
cv_time_balanced = []
for train_idx, valid_idx in skf.split(X_train, y_train):
    X_local_train = preprocessor.fit_transform(X_train.iloc[train_idx])
    y_local_train = y_train.iloc[train_idx].values.ravel()
    X_local_test = preprocessor.transform(X_train.iloc[valid_idx])
    y_local_test = y_train.iloc[valid_idx].values.ravel()

    elapsed_time, roc_auc = fit_predict_imbalanced_model(
        X_local_train, y_local_train, X_local_test, y_local_test)
    cv_time_imbalanced.append(elapsed_time)
    cv_results_imbalanced.append(roc_auc)

    elapsed_time, roc_auc = fit_predict_balanced_model(
        X_local_train, y_local_train, X_local_test, y_local_test)
    cv_time_balanced.append(elapsed_time)
    cv_results_balanced.append(roc_auc)

###############################################################################
# Plot of the results and computation time
###############################################################################

df_results = (pd.DataFrame({'Balanced model': cv_results_balanced,
                            'Imbalanced model': cv_results_imbalanced})
              .unstack().reset_index())
df_time = (pd.DataFrame({'Balanced model': cv_time_balanced,
                         'Imbalanced model': cv_time_imbalanced})
           .unstack().reset_index())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
sns.boxplot(y='level_0', x=0, data=df_time)
sns.despine(top=True, right=True, left=True)
plt.xlabel('time [s]')
plt.ylabel('')
plt.title('Computation time difference using a random under-sampling')

plt.figure()
sns.boxplot(y='level_0', x=0, data=df_results, whis=10.0)
sns.despine(top=True, right=True, left=True)
ax = plt.gca()
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, pos: "%i%%" % (100 * x)))
plt.xlabel('ROC-AUC')
plt.ylabel('')
plt.title('Difference in terms of ROC-AUC using a random under-sampling')
