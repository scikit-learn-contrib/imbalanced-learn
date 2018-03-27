
# coding: utf-8

# ### Do a PCA transformation
# https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/

# In[ ]:

import sys, os
dir = os.path.dirname(os.path.abspath(os.path.realpath('.')))
libRoot = os.path.join(dir, 'imbalanced-learn')
sys.path.insert(0,libRoot)


# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.decomposition import PCA
import time

import re # re.sub() for replacing using regexps
import datetime # ping pong
import multiprocessing # count cpus
from pprint import pprint # beautifully print arrays
import datetime # get current time
import math as math


# ### Setup on new machine
# http://stackoverflow.com/questions/29329667/ipython-notebook-script-deprecated-how-to-replace-with-post-save-hook
# 

# ## Settings

# In[ ]:

n_folds = 5


# In[ ]:

use_ubi = False
if use_ubi:
    import os
    import datetime 
    from ubidots import ApiClient

    # set proxy
    os.environ['http_proxy'] = 'http://proxy.muc:8080' 
    os.environ['https_proxy'] = 'http://proxy.muc:8080'

    # get api and variable
    api = ApiClient(token='O32sAiO8tw4VOTxz24Wmf1IRY7ZoeY')
    ubi_last_timestamp = api.get_variable('5910aee076254222ee1d9d3f')

    new_value = ubi_last_timestamp.save_value({'value': 10, 'context':{'lastTimestamp': "'" + str(datetime.datetime.now()) + "'"}})


# In[ ]:

logpath = "log.log"

class Report():
    @staticmethod
    def getHeader():
        return "\"TS\"" + "," +                "\"TARGET\"" + "," +                "\"DATASET\"" + "," +                "\"MODEL_TYPE\"" + "," +                "\"MODEL_TRAIN_TIME\"" + "," +                "\"MODEL_TRAIN_EVAL_TIME\"" + "," +                "\"MODEL_TEST_TIME\"" + "," +                "\"MODEL_TRAIN_ACCURACY\"" + "," +                "\"MODEL_TRAIN_AUROC\"" + "," +                "\"MODEL_TRAIN_AUPRC\"" + "," +                "\"MODEL_TRAIN_F1\"" + "," +                "\"MODEL_TRAIN_GPERFORMANCE\"" + "," +                "\"MODEL_ACCURACY\"" + "," +                "\"MODEL_AUROC\"" + "," +                "\"MODEL_AUPRC\"" + "," +                "\"MODEL_F1\"" + "," +                "\"MODEL_GPERFORMANCE\"" + "," +                "\"NUM_FEATURES\"" + "," +                "\"NUM_SAMPLE_DATASET\"" + "," +                "\"NUM_SAMPLE_DATASET_POS\"" + "," +                "\"NUM_SAMPLE_DATASET_NEG\"" + "," +                "\"NUM_SAMPLE_TRAIN_BEFORE\"" + "," +                "\"NUM_SAMPLE_TRAIN_BEFORE_POS\"" + "," +                "\"NUM_SAMPLE_TRAIN_BEFORE_NEG\"" + "," +                "\"NUM_SAMPLE_TRAIN_AFTER\"" + "," +                "\"NUM_SAMPLE_TRAIN_AFTER_POS\"" + "," +                "\"NUM_SAMPLE_TRAIN_AFTER_NEG\"" + "," +                "\"BS2\"" + "," +                "\"PROCESS_NAME\"" + "," +                "\"PROCESS_TIME\"" + "," +                "\"PROCESS_NAIVE\"" + "," +                "\"PROCESS_SAMPLING_UP_SMOTE\"" + "," +                "\"PROCESS_SAMPLING_UP_ADASYN\"" + "," +                "\"PROCESS_SAMPLING_DOWN_OSS\"" + "," +                "\"PROCESS_SAMPLING_DOWN_CNN\"" + "," +                "\"PROCESS_SAMPLING_DOWN_TOMEK\"" + "," +                "\"PROCESS_WEIGHT\"" + "," +                "\"PROCESS_SCALE_MINORITY\"" + "," +                "\"PROCESS_SCALE_MODE\"" + "," +                "\"PROCESS_SCALE_TARGET\"" + "," +                "\"PROCESS_SCALE_C\"" + "\r"       
    
    @staticmethod        
    def logToFile(target,
                dataset,
                model_type,
                model_train_time,
                model_train_eval_time,
                model_test_time,
                model_train_accuracy,
                model_train_auroc,
                model_train_auprc,
                model_train_f1,
                model_train_gmean,
                model_accuracy,
                model_auroc,
                model_auprc,
                model_f1,
                model_gmean,
                num_features,
                num_sample_dataset,
                num_sample_dataset_pos,
                num_sample_dataset_neg,
                num_sample_train_before,
                num_sample_train_before_pos,
                num_sample_train_before_meg,
                num_sample_train_after,
                num_sample_train_after_pos, 
                num_sample_train_after_neg,
                bs2,
                process_name,
                process_time,
                process_naive = 0,
                process_sampling_up_smote = 0,
                process_sampling_up_adasyn = 0,
                process_sampling_down_oss = 0,
                process_sampling_down_cnn = 0,
                process_sampling_down_tomek = 0,
                process_weight = 0,
                process_scale_minority = 0,
                process_scale_mode = 0,
                process_scale_target = 0,
                process_scale_c = 0):
        global logpath
        pth = logpath
        import os.path
        if (not os.path.isfile(pth)):
             with open(pth, "a") as myfile:
                    myfile.write(Report.getHeader())

        with open(pth, "a") as myfile:
            myfile.write(Report.getData(target, dataset, model_type, model_train_time, model_train_eval_time, model_test_time, 
                                        model_train_accuracy, model_train_auroc, model_train_auprc, model_train_f1, model_train_gmean,
                                        model_accuracy, model_auroc, model_auprc, model_f1, model_gmean, num_features,
                                 num_sample_dataset, num_sample_dataset_pos, num_sample_dataset_neg,
                                 num_sample_train_before, num_sample_train_before_pos, num_sample_train_before_meg,
                                 num_sample_train_after, num_sample_train_after_pos, num_sample_train_after_neg, bs2,
                                 process_name, process_time, process_naive,
                                 process_sampling_up_smote, process_sampling_up_adasyn, process_sampling_down_oss,
                                 process_sampling_down_cnn, process_sampling_down_tomek, process_weight,
                                 process_scale_minority, process_scale_mode, process_scale_target, process_scale_c)
                        )
    
    @staticmethod        
    def getData(target,
                dataset,
                model_type,
                model_train_time,
                model_train_eval_time,
                model_test_time,
                model_train_accuracy,
                model_train_auprc,
                model_train_auroc,
                model_train_f1,
                model_train_gmean,
                model_accuracy,
                model_auroc,
                model_auprc,
                model_f1,
                model_gmean,
                num_features,
                num_sample_dataset,
                num_sample_dataset_pos,
                num_sample_dataset_neg,
                num_sample_train_before,
                num_sample_train_before_pos,
                num_sample_train_before_meg,
                num_sample_train_after,
                num_sample_train_after_pos, 
                num_sample_train_after_neg,
                bs2,
                process_name,
                process_time,
                process_naive = 0,
                process_sampling_up_smote = 0,
                process_sampling_up_adasyn = 0,
                process_sampling_down_oss = 0,
                process_sampling_down_cnn = 0,
                process_sampling_down_tomek = 0,
                process_weight = 0,
                process_scale_minority = 0,
                process_scale_mode = 0,
                process_scale_target = 0,
                process_scale_c = 0):
        return "\"" + str(datetime.datetime.now()) + "\"" + "," +                "\"" + str(target) + "\"" + "," +                "\"" + str(dataset) + "\"" + "," +                "\"" + str(model_type) + "\"" + "," +                str(model_train_time) + "," +                 str(model_train_eval_time) + "," +                 str(model_test_time) + "," +                   str(model_train_accuracy) + "," +                 str(model_train_auroc) + "," +                    str(model_train_auprc) + "," +                    str(model_train_f1) + "," +                  str(model_train_gmean) + "," +                str(model_accuracy) + "," +                 str(model_auroc) + "," +                    str(model_auprc) + "," +                    str(model_f1) + "," +                  str(model_gmean) + "," +                str(num_features) + "," +                  str(num_sample_dataset) + "," +                  str(num_sample_dataset_pos) + "," +                   str(num_sample_dataset_neg) + "," +                  str(num_sample_train_before) + "," +                   str(num_sample_train_before_pos) + "," +                    str(num_sample_train_before_meg) + "," +                     str(num_sample_train_after) + "," +                           str(num_sample_train_after_pos) + "," +                   str(num_sample_train_after_neg) + "," +                      str(bs2) + "," +                      "\"" + process_name + "\"" + "," +                          str(process_time) + "," +                             str(process_naive) + "," +                     str(process_sampling_up_smote) + "," +                     str(process_sampling_up_adasyn) + "," +                 str(process_sampling_down_oss) + "," +                       str(process_sampling_down_cnn) + "," +                     str(process_sampling_down_tomek) + "," +                  str(process_weight) + "," +                           str(process_scale_minority) + "," +                  "\"" + str(process_scale_mode) + "\"" + "," +                  "\"" + str(process_scale_target) + "\"" + "," +                  str(process_scale_c) + "\r"


# In[ ]:

# define Log function
def log(text, silent=True, force=False):
    if not silent or force:
        print(time.strftime('%Y.%m.%d, %H:%M:%S') + ': ' + text)
    
def ping():
    return datetime.datetime.now()

def pong(dt):
    now = datetime.datetime.now()
    diff = now - dt
    ms = round(diff.total_seconds()*1000)
    return ms

log('init finshed', force=True)


# In[ ]:

def getBS2(X, y):
    tl = TomekLinks()
    X_tl, y_tl = tl.fit_sample(X, y)
    
    num_pos_samples = sum(y)
    num_tomek_links = len(y) - len(y_tl)

    return num_tomek_links / num_pos_samples


# In[ ]:

def indexCategorical(df, columnName):
    df[columnName] = pd.Categorical(df[columnName]).codes
    return df

def renameTargetDropSamePrefix(df, target) :    
    
    rows, colsBefore = df.shape
    
    prefixToDrop = re.sub(r"(.*?)___.*", r"\1___", target)
    log("renaming " + target +         " to \"TARGET\" and dropping all other columns prefixed with " + prefixToDrop)
    df.rename(columns={target: "TARGET"}, inplace = True)
    
    dfReturn = dropPrefix(df, prefixToDrop)
    
    rows, colsAfter = dfReturn.shape
    log("reduced number of columns from {} to {}.".format(colsBefore, colsAfter))
    assert colsAfter < colsBefore
    
    return dfReturn

def dropPrefix(df, prefix) :
    prefix = prefix + ".*"
    log("dropping " + prefix)
    return df.select(lambda x: not re.search(prefix,x), axis = 1)

def testDropPrefix():
    df = pd.DataFrame([
              [1,3,1,0],
              [1,4,1,1],
              [1,5,1,0],
              [1.5,6,1,0],
              [1.7,7,1,0],
              [1,4,1,0],
              [1,6,1,0],
              [1,5,1,1],
              [1,12,1,1],
              [1,9,1,1],
              [1,2,1,1],
              [1,3,1,1],
              [1,5,1,1],
              [2,8,1,0],
              [3,1,0,1],
              [3,2,0,1],
              [4,2,0,0],
              [5,3,0,0]], columns=['PRE1___1', 'PRE1___2', 'PRE2___1','PRE3___1'])

    row, nCol13 = dropPrefix(df, "PRE2___").shape
    row, nCol23 = dropPrefix(df, "PRE1___").shape

    assert nCol13 == 3, "wrong number of cols dropped"
    assert nCol23 == 2, "wrong number of cols dropped"

def pcaFeatureGroup(dfIn, featureGroupPrefix, numberOfDimensionsWhole,
                    numberOfDimensionsTarget, minResultingFeatures = 2):
    """
    e.g. featureGroupPrefix = "RO___"
    """
    rows, cBefore = dfIn.shape
    npMatrix = dfIn.filter(regex = featureGroupPrefix + ".*").as_matrix()
    
    rows, cBeforeOfGroup = npMatrix.shape
    
    if(cBeforeOfGroup == 0):
        log("There are no features belonging to group {}. Returning unmodified DataFrame.".format(featureGroupPrefix))
        return dfIn
    
    # scale
    npMatrix = scale(npMatrix)

    # holds the number of the whole dataset. To be in scale, we first calculate the proportion of the current feature group
    proportion = numberOfDimensionsTarget / numberOfDimensionsWhole
    n_component_target = int(cBeforeOfGroup * proportion)
    
    if (n_component_target < minResultingFeatures):
        n_component_target = minResultingFeatures
    if (cBeforeOfGroup != 0): 
        # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit_transform
        pca = PCA(copy=True, 
                  iterated_power='auto', 
                  n_components=n_component_target, 
                  random_state=None, 
                  svd_solver='auto', 
                  tol=0.0, 
                  whiten=False)
        dfB = dropPrefix(dfIn, featureGroupPrefix)
        rB, cB = dfB.shape
        
        
        npMatrixTransformed = pca.fit_transform(npMatrix)
        dfA = pd.DataFrame(data=npMatrixTransformed[0:,0:], 
                             columns=[featureGroupPrefix + str(num) for num in np.arange(1,npMatrixTransformed.shape[1]+1,1)])#,
                             #index = dfB.index )
        rIn, cIn = dfIn.shape
        rA, cA = dfA.shape
                                 
        dfReturn = np.concatenate([dfA, dfB], axis = 1) # axis 1 is columns, so this direction ->
        dfReturn = pd.DataFrame(data=dfReturn[0:,0:],    # values
                                index=dfA.index,    # 1st column as index
                                columns=dfA.columns.append(dfB.columns))  # 1st row as the column names
        
        rReturn, cReturn = dfReturn.shape
        
        
        log("feature group has been compressed from {} to {} columns".format(cBeforeOfGroup, cA))
        
        assert rReturn == rIn, "num rows of inputted and outputted dataframe do not match"
        assert rIn == rA, "num rows of inputted and PCAd dataframe do not match"
        assert rIn == rB, "num rows of inputted and non PCAd part of initial dataframe do not match"
        assert cReturn == (cB + cA), "concatenating PCAd and non PCAd df into returned df resulted to wrong number of cols"
        assert cBefore == (cB + cBeforeOfGroup), "number of cols from non PCAd and PCAd dataframe before transformation" +                                                  "should add up to initial number of columns"
        
        return dfReturn
    else:
        log("no columns found matching prefix " + featureGroupPrefix + ". Skipping...")
        return dfIn

def print_full(df):
    pd.set_option('display.max_columns', df.shape[1])
    print(df)
    pd.reset_option('display.max_rows')


# ## Read in the different datasources

# In[ ]:


ts = ping()
dfAutomotive = pd.read_csv("in.csv")
nr, ncAutomotive = dfAutomotive.shape
ms = pong(ts)
log("read in dataframe with " + str(nr) + " columns and " + str(ncAutomotive) + " rows in " + str(ms) + "ms", force=True)


# In[ ]:


ts = ping()
dfForest = pd.read_csv("data/forestfires_id.csv")
nr, ncForest = dfForest.shape
ms = pong(ts)
log("read in forest dataframe with " + str(nr) + " columns and " + str(ncForest) + " rows in " + str(ms) + "ms")
#dfForest['area'] = np.log(1 + dfForest['area'])
dfForest['area'] = (dfForest['area'] > 50).astype(bool).astype(int)

nPositive = sum(dfForest['area'])
nNegative = nr-nPositive
log("ratio of forest is {} majority class and {} minority class observations ({:.3f})".format(nNegative, nPositive, nPositive/nNegative))

import matplotlib.pyplot as plt

plt.hist(dfForest['area'], bins=30)
plt.ylabel('Probability')


# In[ ]:


ts = ping()
dfVowel = pd.read_csv("data/vowel-context.csv")
nr, ncVowel = dfVowel.shape
ms = pong(ts)

nPositive = sum(dfVowel['Class'] == 1)
nNegative = nr-nPositive
log("ratio of vowel is {} majority class and {} minority class observations ({:.3f})".format(nNegative, nPositive, nPositive/nNegative))

log("read in vowel dataframe with " + str(nr) + " columns and " + str(ncVowel) + " rows in " + str(ms) + "ms")


# In[ ]:


ts = ping()
dfGlass = pd.read_csv("data/glass.csv")
nr, ncGlass = dfGlass.shape
ms = pong(ts)

sums = []

for i in [1, 2, 3, 5, 6, 7]:
    sums.append(sum(dfGlass['Type'] == i))
    
nPositive = np.round(np.mean(sums))
nNegative = nr-nPositive
log("average ratio of vowel is {} majority class and {} minority class observations ({:.3f})".format(nNegative, nPositive, nPositive/nNegative))

log("read in glass dataframe with " + str(nr) + " columns and " + str(ncGlass) + " rows in " + str(ms) + "ms", force=True)


# In[ ]:

dfPima = pd.read_csv("data/pima.csv") 
nrPima, ncPima = dfPima.shape
nPositive = sum(dfPima['Class'])
nNegative = nrPima-nPositive
log("average ratio of pima is {} majority class and {} minority class observations ({:.3f})".format(nNegative, nPositive, nPositive/nNegative))
log("features={}".format(ncPima))

dfPhoneme = pd.read_csv("data/phoneme.csv") 
nrPhoneme, ncPhoneme = dfPhoneme.shape
nPositive = sum(dfPhoneme['class'])
nNegative = nrPhoneme-nPositive
log("average ratio of phoneme is {} majority class and {} minority class observations ({:.3f})".format(nNegative, nPositive, nPositive/nNegative))

dfVehicle = pd.read_csv("data/vehicle.csv") 
nrVehicle, ncVehicle = dfVehicle.shape

sums = []
for i in [1, 2, 3]:
    sums.append(sum(dfVehicle['TARGET'] == i))
    
nPositive = np.round(np.mean(sums))
nNegative = nrVehicle-nPositive
log("average ratio of vehicle is {} majority class and {} minority class observations ({:.3f})".format(nNegative, nPositive, nPositive/nNegative))

dfAbalone = pd.read_csv("data/abalone_9_18.csv") 
nrAbalone, ncAbalone = dfAbalone.shape
nPositive = sum(dfAbalone['Rings'])
nNegative = nrAbalone-nPositive
log("ratio of abalone is {} majority class and {} minority class observations ({:.3f})".format(nNegative, nPositive, nPositive/nNegative))

dfSatimage = pd.read_csv("data/satimage.csv") 
nrSatimage, ncSatimage = dfSatimage.shape

sums = []
for i in [1, 2, 3, 4, 5, 7]:
    sums.append(sum(dfSatimage['CLASS'] == i))
nPositive = np.round(np.mean(sums))
nNegative = nrSatimage-nPositive
log("average ratio of satimage is {} majority class and {} minority class observations ({:.3f})".format(nNegative, nPositive, nPositive/nNegative))

dfMammography = pd.read_csv("data/mammography.csv") 
nrMammography, ncMammography = dfMammography.shape
nPositive = sum(dfMammography['target'])
nNegative = nrMammography-nPositive
log("ratio of mammography is {} majority class and {} minority class observations ({:.3f})".format(nNegative, nPositive, nPositive/nNegative))


# In[ ]:

from sklearn.preprocessing import scale

def train_test_split_index(X, y, index):
    """
    split X and y into training and testing based on index
    """
    X_scaled = scale(X)
    
    if not isinstance(y, np.ndarray):
        y = y.as_matrix()
    if not isinstance(X_scaled, np.ndarray):
        X_scaled = X_scaled.as_matrix()
    if not isinstance(index, np.ndarray):
        index = index.as_matrix()
    
    X_train = X_scaled[index == 0]
    X_test = X_scaled[index == 1]
    y_train = y[index == 0]
    y_test = y[index == 1]
    

    
    return X_train, X_test, y_train, y_test

def train_test_split_scaled(X, y):
    """
    split X and y into training and testing based on index
    """
    X_scaled = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
    
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.as_matrix()
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.as_matrix()
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.as_matrix()
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.as_matrix()
    
    return X_train, X_test, y_train, y_test


# In[ ]:

def getDataFrameForTarget(df, target = "DTC___1196802", prefixesToDrop = ["BEFUND___", "DK___"]) :
    log("getting dataframe for target " + target + " while dropping " + str(prefixesToDrop))
    # select target and all other columns except columns with the same prefix
    dfTemp = renameTargetDropSamePrefix(df, target)

    # get DTCs
    #dfDTC2 = df.filter(regex=("(META|CP|RO|DTC|EE|SC|MV)___.*")) # doesn't work
    for prefix in prefixesToDrop: 
        dfTemp = dropPrefix(dfTemp, prefix)

    # convert Categories to Indexes
    dfTemp = indexCategorical(dfTemp, "META___CARID")
    
    return dfTemp
    
# do PCA for every featuregroup separately
def doPCA(df, featureGroupsPCA = ["CP", "RO", "EE", "MV", "SC", "DTC"], numberOfDimensionsTarget = 100):
    ## no meta
    assert isinstance(df, pd.DataFrame), "dataframe needs to be a pandas.DataFrame to allow filtering for " +                                         "different feature groups."
    dfPCA = df
    dummy, colsBefore = dfPCA.shape
    for group in featureGroupsPCA:
        log("Working group " + group + "...")
        dfPCA = pcaFeatureGroup(dfPCA, featureGroupPrefix = group + "___", numberOfDimensionsWhole = colsBefore,
                                numberOfDimensionsTarget = numberOfDimensionsTarget)
        
    dummy, colsAfter = dfPCA.shape
    log("reduced dimensions from {} to {} using PCA.".format(colsBefore, colsAfter))
    
    return dfPCA


# # create the datasets according to the sampling strategy
# 
# Variants inlcude:
# 1. a naive approach to serve as baseline
# 2. one-sided selection
# 3. condest nearest neighbour
# 4. SMOTE
# 5. assigning costs
# 6. preferably sample same cars
# 7. use heuristic to identify most valuable majority-class observations

# In[ ]:


from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from imblearn.scaling import CSS 
from imblearn.under_sampling import TomekLinks, OneSidedSelection, CondensedNearestNeighbour
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler 
    
def createDatasetXY(df, indexFeatureStart = 0, indexFeatureEnd = -1, targetColumnName = "TARGET"):
    """
    creates train / test splits for X and y using the given dataframe
    
    :param df: The dataframe to be used
    :param indexFeatureStart: Index to start selecting features
    :param indexFeatureEnd: Index to stop selecting features
    :return: X, y
    """
    
    if(indexFeatureEnd == -1):
        indexFeatureEnd = len(df.columns) - 4
    X = df.ix[:,indexFeatureStart:indexFeatureEnd] # 3 meta cols (META___{RANDOM, CARID, PLANNED}) and the target row
    y = df[targetColumnName].astype(bool).astype(int)

    return X, y

def shuffleTwo(a, b):
    a1 = pd.DataFrame(a)
    indexes = a1.index # get the indices, the first magical column
    assert len(a) == len(b), "lenth of a ({}) doesn't match length of b ({})".format(len(a), len(b))
    p = np.random.permutation(len(indexes))
    indexesShuffled = indexes[p]
    a2, b2 = a1.ix[indexesShuffled], b[indexesShuffled]
    
    return a2, b2

def createDatasetUsingMetaXy(df, indexFeatureStart = 0, indexFeatureEnd = -1, 
                           targetColumnName = "TARGET",
                           metaColumnName = "META___PLANNED"):
    # as majority observations such examples will be selected, where the metaColumnName is 1
    
    if(indexFeatureEnd == -1):
        indexFeatureEnd = len(df.columns) - 4
    X = df.ix[:,indexFeatureStart:indexFeatureEnd] # 3 meta cols (META___{RANDOM, CARID, PLANNED}) and the target row
    prefix = re.sub(r'___.*', r'', metaColumnName)
    y = df[targetColumnName].astype(bool).astype(int)
    
    # select only the "META == 1" or "TARGET == 1" rows.
    indices = y[y > 0].index
    indices = indices.append(X[X[metaColumnName] > 0].index)
    indices = np.unique(indices)
    X = X.loc[indices]
    X = dropPrefix(X, prefix)   
    y = y.loc[indices]
    
    return X, y

def createDatasetUsingRandomXy(df, indexFeatureStart = 0, indexFeatureEnd = -1, 
                               targetColumnName = "TARGET", ratio = 0.1):
    # ratio defines the ratio between majority:minority (10 means: 10 times as much majority)
    # as majority observations such examples will be selected, where the metaColumnName is 1
    
    if(indexFeatureEnd == -1):
        indexFeatureEnd = len(df.columns) - 4
    X = df.ix[:,indexFeatureStart:indexFeatureEnd] # 3 meta cols (META___{RANDOM, CARID, PLANNED}) and the target row
    y = df[targetColumnName].astype(bool).astype(int)
    
    
    log('Original dataset shape {}'.format(Counter(y)))

    targetRatio = 1/ratio
    num_pos = sum(y)
    currentRatio = num_pos / (len(y)-num_pos)
    log("Current ratio={}, targetRatio={}".format(currentRatio, targetRatio))
    if(targetRatio < currentRatio):
        targetRatio = currentRatio
    
    rus = RandomUnderSampler(random_state=42, ratio = targetRatio)
    X_res, y_res = rus.fit_sample(X, y)
    log('Resampled dataset shape {}'.format(Counter(y_res)))
    
    X_res = pd.DataFrame(data=X_res[0:,0:], # values
                         #index=X.index,     # 1st column as index
                         columns=X.columns)
    
    log("class returned by RandomXy for X is {}".format(str(type(X_res))))
    
    return X_res, y_res

def createNaiveDataset(X_train, y_train):
    """
    creates train / test splits for X and y using the given dataframe
    
    :param df: The dataframe to be used
    :param indexFeatureStart: Index to start selecting features
    :param indexFeatureEnd: Index to stop selecting features
    :return: X_train, X_test, y_train, y_test used for training
    """
    log("creating dataset [naive mode]...")
    
    return X_train, y_train

def createSMOTEDataset(X_train, y_train):
    log("creating dataset [SMOTE]...")
    
    log('Original dataset shape {}'.format(Counter(y_train)))

    # SMOTE expects n_neighbors <= n_samples
    n_neighbors = 5
    n_samples = sum(y_train)
    n_samples_total, dummy = X_train.shape
    # bug in sklearn\neighbors\base.py in kneighbors(self, X, n_neighbors, return_distance)
    # that causes the number of samples to be 1 smaller.
    if ((n_samples-1) < n_neighbors):
        log("reducing n_neighbors ({}) to number of samples ({})".format(n_neighbors, n_samples))
        n_neighbors = n_samples - 1  
    if ((n_samples_total-1) < n_neighbors):
        log("reducing n_neighbors ({}) to total number of samples ({})".format(n_neighbors, n_samples_total))
        n_neighbors = n_samples_total - 1
    
    sm = SMOTE(random_state=42, k_neighbors=n_neighbors)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    log('Resampled dataset shape {}'.format(Counter(y_train_res)))
    
    # shuffle
    X_train_res, y_train_res = shuffleTwo(X_train_res, y_train_res)
    
    return X_train_res, y_train_res

def createADASYNDataset(X_train, y_train):
    from imblearn.over_sampling import ADASYN 
    ada = ADASYN()
    X_train_res, y_train_res = ada.fit_sample(X_train, y_train)
    
    # shuffle
    X_train_res, y_train_res = shuffleTwo(X_train_res, y_train_res)
    
    return X_train_res, y_train_res

def createTomekDataset(X_train, y_train):
    tl = TomekLinks(return_indices=True)
    X_train_res, y_train_res, idx_resampled = tl.fit_sample(X_train, y_train)
    
    # shuffle
    X_train_res, y_train_res = shuffleTwo(X_train_res, y_train_res)
    
    return X_train_res,  y_train_res

def createOSSDataset(X_train, y_train):
    oss = OneSidedSelection(return_indices=True)
    X_train_res, y_train_res, idx_resampled = oss.fit_sample(X_train, y_train)
    
    # shuffle
    X_train_res, y_train_res = shuffleTwo(X_train_res, y_train_res)
#     X_train_res = pd.DataFrame(X_train_res)
#     assert len(X_train_res) == len(y_train_res)
#     p = np.random.permutation(len(X_train_res))
#     X_train_res.reset_index(drop=True)
#     X_train_res, y_train_res = X_train_res.ix[p], y_train_res[p]

    return X_train_res, y_train_res

def createCNNDataset(X_train, y_train):
    cnn = CondensedNearestNeighbour(return_indices=True)
    X_train_res, y_train_res, idx_resampled = cnn.fit_sample(X_train, y_train)
    
    # shuffle
    X_train_res, y_train_res = shuffleTwo(X_train_res, y_train_res)
    
    return X_train_res, y_train_res

def createScaledDataset(X_train, y_train, targetClass = "majority", c = 0.2, mode = "constant", 
                        verbose = False):
    
    css = CSS(mode=mode, target=targetClass, c=c, shuffle=True)
    return css.fit_sample(X_train,y_train) # X_s, y_s

def testVisualCreateScaledDataset():
    iVowel = 0
    dfVowelSub = dfVowel.copy()
    dfVowelSub['Class'] = (dfVowelSub['Class'] == iVowel).astype(bool)
    dfVowelSub['Class'] = dfVowelSub['Class'].astype(int)

    XVowel, yVowel = createDatasetXY(df = dfVowelSub, indexFeatureStart = 1, 
                                     indexFeatureEnd = ncVowel-2, targetColumnName = "Class")
    zVowel = dfVowel['Train or Test']
    X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split_index(XVowel, yVowel, zVowel)     

    X_train, X_test, y_train, y_test = createNaiveDataset(X_train_pre, X_test_pre, y_train_pre, y_test_pre)
    retVal = trainLR(X_train, X_test, y_train, y_test, balanced=None, scoring="auROC")

    pca = PCA(n_components=2)
    pcaFitted = pca.fit(X_train)
    X_r = pcaFitted.transform(X_train)
    plt.scatter(X_r[:,0], X_r[:,1], c=y_train, alpha=0.5)
    #np.savetxt("out/visualize/vowel_class_1/no_scale.csv", np.column_stack([X_r[:,0], X_r[:,1], y_train]), delimiter=",")
    plt.show()

    X_train, X_test, y_train, y_test = createScaledDataset(X_train_pre, X_test_pre, y_train_pre, y_test_pre, c = 0.3)
    retVal = trainLR(X_train, X_test, y_train, y_test, balanced=None, scoring="auROC")
    pca = PCA(n_components=2)
    X_r = pcaFitted.transform(X_train)
    plt.scatter(X_r[:,0], X_r[:,1], c=y_train, alpha=0.5)
    #np.savetxt("out/visualize/vowel_class_1/scale_0.2.csv", np.column_stack([X_r[:,0], X_r[:,1], y_train]), delimiter=",")
    plt.show()

def testCreateScaledDataset():
    df = pd.DataFrame([
              [1,3,1,0],
              [1,4,1,1],
              [1,5,1,1],
              [2,8,1,0],
              [3,1,0,1],
              [3,2,0,1],
              [4,2,0,0],
              [5,3,0,0]], columns=['a', 'b', 'target','train'])
    dfTrainTest = df['train']
    X, y = createDatasetXY(df, indexFeatureStart = 0, indexFeatureEnd = 2, targetColumnName = "target")
    X_train, X_test, y_train, y_test = train_test_split_index(X, y, dfTrainTest)
    X_train
    c = 0.3
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = createScaledDataset(X_train, X_test, y_train, 
                                                                                       y_test, mode = "single", c = c)
    
    
    (X_train.ix[6,0] + X_train.ix[7,0])/2*c + (1-c) * X_train.ix[6,0] == X_train_scaled.ix[6,0]
    (X_train.ix[6,1] + X_train.ix[7,1])/2*c + (1-c) * X_train.ix[7,1] == X_train_scaled.ix[7,1]
    X_train.ix[0,1] == X_train_scaled.ix[0,1]
    X_train.ix[3,1] == X_train_scaled.ix[3,1]


# In[ ]:

from sklearn import datasets, neighbors, linear_model, svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, make_scorer, auc, precision_recall_curve
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

lastModel = 0
lastY = 0
lastPred = 0

class trainResult():
    auROC = -1
    auPRC = -1
    accuracy = -1
    f1 = -1
    gmean = -1
    train_time = -1
    test_time = -1
    gmean = -1
    train_eval_time = -1
    train_accuracy = -1
    train_auroc = -1
    train_auprc = -1
    train_f1 = -1
    train_gmean = -1
    ms_process = -1

def get_au_prc(real, preds, pos_label=1):
    precision, recall, _ = precision_recall_curve(real, preds, pos_label=pos_label)
    auPRC = metrics.auc(precision, recall, reorder=True)
    
    return auPRC
    
def getMetrics(y = np.array([1, 1, 2, 2]), pred = np.array([0.1, 0.4, 0.35, 0.8]), threshold = 0.5):
    
    if (len(np.unique(pred)) < 2) : 
        #log("all predictions the same. setting auc to 0 and f1 to 0.")
        accuracy = metrics.accuracy_score(y, pred > threshold)
        return (0, 0, accuracy, 0, 0)
    
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr) # AU ROC
        f1 = metrics.f1_score(y, pred > threshold)
        accuracy = metrics.accuracy_score(y, pred > threshold)
        precision = metrics.precision_score(y, pred > threshold)
        recall = metrics.recall_score(y, pred > threshold)
        auPRC = get_au_prc(y, pred, pos_label=1)
        
        g_mean = math.sqrt(precision * recall)
        return (auc, f1, accuracy, g_mean, auPRC)

def getNumberOfTomekLinks(X, y):
    from imblearn.under_sampling import TomekLinks
    tl = TomekLinks(return_indices=True)
    n_row_before, dummy = X.shape
    X_resampled, y_resampled, idx_resampled = tl.fit_sample(X, y)
    n_row_after, dummy = X_resampled.shape
    tlFound = n_row_before - n_row_after
    log(str(tlFound) + " tomek links found")
    return tlFound

from sklearn.model_selection import cross_val_score
def getCVPerformanceOld(clf, X_train, y_train, scoring = "roc_auc"):
    
    if len(np.unique(y_train)) != 2:
        return 0
    
    np.set_printoptions(threshold=np.inf)
    pprint(y_train)
    
    # for more scorers see http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scores = cross_val_score(clf, X_train, y_train, cv = n_folds, scoring = scoring)
            
    return np.mean(scores)
    
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
def getCVPerformance(clf, X_train, y_train, scoring = "roc_auc"):
    
    if len(np.unique(y_train)) != 2:
        return 0
    
    pred = cross_val_predict(clf, X_train, y_train, cv=n_folds)
 
    return roc_auc_score(y_train, pred)

def trainNNScale(X_train, X_test, y_train, y_test,
                 c_scale = 0, mode = "constant", targetClass = "minority"):
    log("training NN")
    retVal = trainResult()
    ms_process_total = 0

    
    # train    
    pTrain = ping()
    X_train_pre = pd.DataFrame(X_train).reset_index(drop=True).as_matrix() # otherwise indices from X will be used
    X_test = pd.DataFrame(X_test).reset_index(drop=True).as_matrix().astype(float) # otherwise indices from X will be used
            
            
    if (str(type(y_train)) != "<class 'numpy.ndarray'>"):
        y_train_pre = y_train.reset_index(drop=True) # otherwise indices from X will be used
    else:
         y_train_pre = y_train
            
    if (str(type(y_test)) != "<class 'numpy.ndarray'>"):
        y_test = y_test.reset_index(drop=True).as_matrix() # otherwise indices from X will be used
    from sklearn.model_selection import KFold
    bestAuROC = 0
    bestAuPRC = 0
    bestAccuracy = 0
    bestGmean = 0
    bestF1 = 0
    bestSolver = 'lbfgs'
    bestActivation = 'relu'
    solverz = ['lbfgs', 'sgd', 'adam']
    activationz = ['identity', 'logistic', 'tanh', 'relu']
    layerz = [(2,2), (5,2), (5,5), (10,5), (10,10), (2,2,2), (5,5,5), (10,10,10)]
    for solver in solverz:
        for activation in activationz:
            for layer in layerz:
                kf = KFold(n_splits=n_folds)
                kf.get_n_splits(X_train_pre)

                scoresAuROC = []
                scoresF1 = []
                scoresAccuracy = []
                scoresGmean = []
                scoresAuPRC = []

                for train_index, test_index in kf.split(X_train):
                    X_train_cv, X_test_cv = X_train_pre[train_index], X_train_pre[test_index]
                    y_train_cv, y_test_cv = y_train_pre[train_index], y_train_pre[test_index]

                    clf = MLPClassifier(activation  = activation, solver=solver, alpha=1e-5,
                                        hidden_layer_sizes=layer, random_state=1)                             

                    # tests will be unaffected
                    if (c_scale > 0):
                        pProcess = ping()
                        X_train_cv, y_train_cv = createScaledDataset(X_train_cv, y_train_cv, mode=mode, c=c_scale, targetClass=targetClass)
                        ms_process_total += pong(pProcess)                    

                    if ((np.isnan(X_train_cv)).any):
                        X_train_cv = np.nan_to_num(X_train_cv)

                    # train
                    clf.fit(X_train_cv, y_train_cv)
                    pred = clf.predict(X_test_cv)

                    # eval
                    auROC, f1, accuracy, gmean, auPRC = getMetrics(y_test_cv, pred)

                    scoresAuROC.append(auROC)
                    scoresF1.append(f1)
                    scoresAccuracy.append(accuracy)
                    scoresGmean.append(gmean)
                    scoresAuPRC.append(auPRC)

                meanScoreAuROC = np.mean(scoresAuROC)
                meanScoreF1 = np.mean(scoresF1)
                meanScoreAccuracy = np.mean(scoresAccuracy)
                meanScoreGmean = np.mean(scoresGmean)
                meanScoreAuPRC = np.mean(scoresAuPRC)

                if(meanScoreAuPRC > bestAuPRC):
                    bestSolver = solver
                    bestActivation = activation
                    bestAuROC = auROC
                    bestAccuracy = meanScoreAccuracy
                    bestGmean = meanScoreGmean
                    bestF1 = meanScoreF1
                    bestAuPRC = meanScoreAuPRC
                
    if (c_scale > 0):
        log("scaling final train data...")
        pProcess = ping()
        X_train, y_train = createScaledDataset(X_train, y_train,
                                               c=c_scale, mode=mode, targetClass=targetClass)
        ms_process_total /= len(solverz)
        ms_process_total /= len(activationz)
        ms_process_total = pong(pProcess)
        
    if((np.isnan(X_train)).any):
        X_train= np.nan_to_num(X_train)
        
    
    clf = MLPClassifier(activation  = bestActivation, solver=bestSolver, alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train_cv, y_train_cv)            
    retVal.train_time = pong(pTrain)
    
    # get CV train metrics
    pTrainCV = ping()
    bestAuROC = bestAuROC
    bestAuPRC = bestAuPRC
    bestAccuracy, bestF1, bestGmean = -1, -1, -1
    retVal.train_eval_time = pong(pTrainCV)
    retVal.train_accuracy = bestAccuracy
    retVal.train_auroc = bestAuROC
    retVal.train_auprc = bestAuPRC
    retVal.train_f1 = bestF1
    retVal.train_gmean = bestGmean
    if(c_scale > 0):
        retVal.ms_process = ms_process_total
    
    pred = clf.predict(X_test)
    
    pTest = ping()
    retVal.auROC, retVal.f1, retVal.accuracy, retVal.gmean, retVal.auPRC = getMetrics(y_test, pred)
    retVal.test_time = pong(pTest)

    log('NN score: auROC={}f, auPRC={}f, f1={}, accuracy={}, gmean={}'.format(retVal.auROC, retVal.auPRC, retVal.f1, retVal.accuracy, retVal.gmean))
    
    return retVal
    

def trainOCC(X_train, X_test, y_train, y_test):
    
    #TODO : Implement cross validation http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
    
    log("training OCC")
    retVal = trainResult()
    
    pTrain = ping()
    
    X_train_pre = pd.DataFrame(X_train).reset_index(drop=True).as_matrix() # otherwise indices from X will be used
    X_test = pd.DataFrame(X_test).reset_index(drop=True).as_matrix().astype(float) # otherwise indices from X will be used
            
            
    if (str(type(y_train)) != "<class 'numpy.ndarray'>"):
        y_train_pre = y_train.reset_index(drop=True) # otherwise indices from X will be used
    else:
         y_train_pre = y_train
            
    if (str(type(y_test)) != "<class 'numpy.ndarray'>"):
        y_test = y_test.reset_index(drop=True).as_matrix() # otherwise indices from X will be used
            
    bestAuROC = 0
    bestAuPRC = 0
    bestNu = 0.1
    bestKernel = 'linear'
    bestGamma = 0.1
    for gamma in [0.001, 0.01, 0.1, 1]:
        for nu in [0.01, 0.1, 0.5, 0.75, 1]:
            for kernel in ['linear', 'poly', 'sigmoid']: # 'rbf'
                #log("CV for gamma={}, nu={}, kernel={}".format(gamma, nu, kernel))

                kf = KFold(n_splits=n_folds)
                kf.get_n_splits(X_train_pre)
                # print(kf) # print info about folds

                scoresAuROC = []
                scoresF1 = []
                scoresAccuracy = []
                scoresGmean = []               
                scoresAuPRC = []
                
                for train_index, test_index in kf.split(X_train):
                    X_train_cv, X_test_cv = X_train_pre[train_index], X_train_pre[test_index]
                    y_train_cv, y_test_cv = y_train_pre[train_index], y_train_pre[test_index]

                    clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, tol = 0.01)
                    #n_estimators = 10
                    #clf = BaggingClassifier(svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma),
                    #                        max_samples=1.0 / n_estimators, n_estimators=n_estimators)

                    #pprint(X_train_cv[y_train_cv == 1])
                    #np.savetxt("occ_data.csv", X_train_cv, delimiter=",")
                    
                    nrow, ncol = X_train_cv[y_train_cv == 1].shape
                    
                    if (nrow == 0):
                        log("no samples (CV for gamma={}, nu={}, kernel={}), continueing...".format(gamma, nu, kernel))
                        continue
                    
                    clf.fit(X_train_cv[y_train_cv == 1])
                    pred = clf.predict(X_test_cv)
                    pred[pred < 0] = 0 # SVM outputs -1 for the "0" class
                    
                    # eval
                    auROC, f1, accuracy, gmean, auPRC = getMetrics(y_test_cv, pred)
                    
                    scoresAuROC.append(auROC)
                    scoresF1.append(f1)
                    scoresAccuracy.append(accuracy)
                    scoresGmean.append(gmean)
                    scoresAuPRC.append(auPRC)
                    
                meanScoreAuROC = np.mean(scoresAuROC)
                meanScoreAuPRC = np.mean(scoresAuRRC)
                meanScoreF1 = np.mean(scoresF1)
                meanScoreAccuracy = np.mean(scoresAccuracy)
                meanScoreGmean = np.mean(scoresGmean)
                
                if(meanScoreAuPRC > bestAuPRC):
                    bestNu = nu
                    bestGamma = gamma
                    bestKernel = kernel
                    bestAuROC = auROC
                    bestAuPRC = auPRC
                    bestAccuracy = meanScoreAccuracy
                    bestGmean = meanScoreGmean
                    bestF1 = meanScoreF1
                 
#     retVal.train_eval_time = 0
#     retVal.train_accuracy = bestAccuracy
#     retVal.train_auc = bestAuROC
#     retVal.train_f1 = bestF1
#     retVal.train_gmean = bestGmean
    
    log('CV finished. Achieved best auROC={} using nu={}, gamma={} and kernel={}'.format(bestAuROC, bestNu, 
                                                                                           bestGamma, bestKernel))
    clf = svm.OneClassSVM(nu=bestNu, kernel=bestKernel, gamma=bestGamma)
    clf.fit(X_train[y_train == 1]) # final training using all data
    retVal.train_time = pong(pTrain)
    
    # get CV train metrics
    pTrainCV = ping()
    bestAuROC = getCVPerformance(clf, X_train, y_train)
    bestAccuracy, bestF1, bestGmean = -1, -1, -1
    retVal.train_eval_time = pong(pTrainCV)
    retVal.train_accuracy = bestAccuracy
    retVal.train_auroc = bestAuROC
    retVal.train_auprc = bestAuPRC
    retVal.train_f1 = bestF1
    retVal.train_gmean = bestGmean
    
    pTest = ping()
    pred = clf.predict(X_test)
    pred[pred < 0] = 0 # SVM outputs -1 for the "0" class
    
    retVal.auROC, retVal.f1, retVal.accuracy, retVal.gmean, retVal.auPRC = getMetrics(y_test, pred)
    retVal.test_time = pong(pTest)
    
    log('OCC score: auROC={}f, auPRC={}f, f1={}, accuracy={}, gmean={}'.format(retVal.auROC, retVal.auPRC, retVal.f1, retVal.accuracy, retVal.gmean))
    #log('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))
    
    return retVal


def testTrainOCC():
    df = pd.DataFrame([
              [1,3,1,0],
              [1,4,1,1],
              [1,5,1,0],
              [1.5,6,1,0],
              [1.7,7,1,0],
              [1,4,1,0],
              [1,6,1,0],
              [1,5,1,1],
              [1,12,1,1],
              [1,9,1,1],
              [1,2,1,1],
              [1,3,1,1],
              [1,5,1,1],
              [2,8,1,0],
              [3,1,0,1],
              [3,2,0,1],
              [4,2,0,0],
              [5,3,0,0]], columns=['a', 'b', 'target','train'])
    dfTrainTest = df['train']
    X, y = createDatasetXY(df, indexFeatureStart = 0, indexFeatureEnd = 2, targetColumnName = "target")
    X_train, X_test, y_train, y_test = train_test_split_index(X, y, dfTrainTest)

    trainOCC(X_train, X_test, y_train, y_test)
    
def trainOCCScale(X_train, X_test, y_train, y_test,
                  c_scale = 0, mode = "constant", targetClass = "minority"):
    log("training OCC")
    retVal = trainResult()

    pTrain = ping()

    X_train_pre = pd.DataFrame(X_train).reset_index(drop=True).as_matrix()  # otherwise indices from X will be used
    X_test = pd.DataFrame(X_test).reset_index(drop=True).as_matrix().astype(
        float)  # otherwise indices from X will be used

    if (str(type(y_train)) != "<class 'numpy.ndarray'>"):
        y_train_pre = y_train.reset_index(drop=True)  # otherwise indices from X will be used
    else:
        y_train_pre = y_train

    if (str(type(y_test)) != "<class 'numpy.ndarray'>"):
        y_test = y_test.reset_index(drop=True).as_matrix()  # otherwise indices from X will be used

    ms_process_total = 0
    bestAuROC = 0
    bestAuPRC = 0
    bestNu = 0.1
    bestKernel = 'linear'
    bestGamma = 0.1
    gammaz = [0.001, 0.01, 0.1, 1]
    nuz = [0.01, 0.1, 0.5, 0.75, 1]
    kernelz =  ['linear', 'poly', 'sigmoid']
    for gamma in gammaz:
        for nu in nuz:
            for kernel in kernelz:  # 'rbf'
                # log("CV for gamma={}, nu={}, kernel={}".format(gamma, nu, kernel))

                kf = KFold(n_splits=n_folds)
                kf.get_n_splits(X_train_pre)
                # print(kf) # print info about folds

                scoresAuROC = []
                scoresAuPRC = []
                scoresF1 = []
                scoresAccuracy = []
                scoresGmean = []
                scoresAuPRC = []

                for train_index, test_index in kf.split(X_train):
                    X_train_cv, X_test_cv = X_train_pre[train_index], X_train_pre[test_index]
                    y_train_cv, y_test_cv = y_train_pre[train_index], y_train_pre[test_index]

                    # tests will be unaffected
                    if (c_scale > 0):
                        pProcess = ping()
                        X_train_cv, y_train_cv = createScaledDataset(X_train_cv, y_train_cv, mode=mode, c=c_scale,
                                                                     targetClass=targetClass)
                        ms_process_total += pong(pProcess)

                    if ((np.isnan(X_train_cv)).any):
                        X_train_cv = np.nan_to_num(X_train_cv)

                    clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, tol=0.01)
                    # n_estimators = 10
                    # clf = BaggingClassifier(svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma),
                    #                        max_samples=1.0 / n_estimators, n_estimators=n_estimators)

                    # pprint(X_train_cv[y_train_cv == 1])
                    # np.savetxt("occ_data.csv", X_train_cv, delimiter=",")

                    nrow, ncol = X_train_cv[y_train_cv == 1].shape

                    if (nrow == 0):
                        log("no samples (CV for gamma={}, nu={}, kernel={}), continueing...".format(gamma, nu, kernel))
                        continue

                    clf.fit(X_train_cv[y_train_cv == 1])
                    pred = clf.predict(X_test_cv)
                    pred[pred < 0] = 0  # SVM outputs -1 for the "0" class

                    # eval
                    auROC, f1, accuracy, gmean, auPRC = getMetrics(y_test_cv, pred)

                    scoresAuROC.append(auROC)
                    scoresF1.append(f1)
                    scoresAccuracy.append(accuracy)
                    scoresGmean.append(gmean)
                    scoresAuPRC.append(auPRC)

                meanScoreAuROC = np.mean(scoresAuROC)
                meanScoreF1 = np.mean(scoresF1)
                meanScoreAccuracy = np.mean(scoresAccuracy)
                meanScoreGmean = np.mean(scoresGmean)
                meanScoreAuPRC = np.mean(scoresAuPRC)
                
                if(meanScoreAuPRC > bestAuPRC):
                    bestAuROC = auROC
                    bestAccuracy = meanScoreAccuracy
                    bestGmean = meanScoreGmean
                    bestF1 = meanScoreF1
                    bestAuPRC = meanScoreAuPRC
                    bestNu = nu
                    bestKernel = kernel
                    bestGamma = gamma

                    #     retVal.train_eval_time = 0
                    #     retVal.train_accuracy = bestAccuracy
                    #     retVal.train_auc = bestAuROC
                    #     retVal.train_f1 = bestF1
                    #     retVal.train_gmean = bestGmean

    log('CV finished. Achieved best auROC={} using nu={}, gamma={} and kernel={}'.format(bestAuROC, bestNu,
                                                                                           bestGamma, bestKernel))

    if (c_scale > 0):
        log("scaling final train data...")
        pProcess = ping()
        X_train, y_train = createScaledDataset(X_train, y_train,
                                                     c = c_scale, mode = mode, targetClass = targetClass)
        ms_process_total /= len(gammaz) # allowed, because: Folds can be calculated out of the CV loop.
        ms_process_total /= len(nuz) # allowed, because: Folds can be calculated out of the CV loop.
        ms_process_total /= len(kernelz) # allowed, because: Folds can be calculated out of the CV loop.
        ms_process_total = pong(pProcess)

    if((np.isnan(X_train)).any):
        X_train= np.nan_to_num(X_train)
        
    clf = svm.OneClassSVM(nu=bestNu, kernel=bestKernel, gamma=bestGamma)
    clf.fit(X_train[y_train == 1])  # final training using all data
    retVal.train_time = pong(pTrain)

    # get CV train metrics
    pTrainCV = ping()
    bestAuROC = bestAuROC
    bestAccuracy, bestF1, bestGmean = -1, -1, -1
    retVal.train_eval_time = pong(pTrainCV)
    retVal.train_accuracy = bestAccuracy
    retVal.train_auc = bestAuROC
    retVal.train_f1 = bestF1
    retVal.train_gmean = bestGmean
    if(c_scale > 0):
        retVal.ms_process = ms_process_total
        
    pTest = ping()
    pred = clf.predict(X_test)
    pred[pred < 0] = 0  # SVM outputs -1 for the "0" class

    retVal.auROC, retVal.f1, retVal.accuracy, retVal.gmean, retVal.auPRC = getMetrics(y_test, pred)
    retVal.test_time = pong(pTest)

    log('OCC score: auROC={}f, auPRC={}f, f1={}, accuracy={}, gmean={}'.format(retVal.auROC, retVal.auPRC, 
                                                                               retVal.f1, retVal.accuracy, retVal.gmean))
    

    return retVal

def trainKNN(X_train, X_test, y_train, y_test):
    log("training KNN")
    retVal = trainResult()
    
    knn = neighbors.KNeighborsClassifier()
    
    pTrain = ping()
    model = knn.fit(X_train, y_train)
    retVal.train_time = pong(pTrain)
    
    # get CV train metrics
    pTrainCV = ping()
    bestAuROC = getCVPerformance(model, X_train, y_train)
    bestAccuracy, bestF1, bestGmean = -1, -1, -1
    retVal.train_eval_time = pong(pTrainCV)
    retVal.train_accuracy = bestAccuracy
    retVal.train_auc = bestAuROC
    retVal.train_f1 = bestF1
    retVal.train_gmean = bestGmean
    
    pred = model.predict(X_test)
    
    lastModel = model
    lastY = y_test
    lastPred = pred
    
    pTest = ping()
    retVal.auROC, retVal.f1, retVal.accuracy, retVal.gmean, retVal.auPRC = getMetrics(y_test, pred)
    retVal.test_time = pong(pTest)

    log('KNN score: auROC={}f, auPRC={}f, f1={}, accuracy={}, gmean={}'.format(retVal.auROC, retVal.auPRC, retVal.f1, retVal.accuracy, retVal.gmean))
    
    return retVal
    
from sklearn import datasets, neighbors, linear_model, svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold, cross_val_score

def trainKNNScale(X_train, X_test, y_train, y_test,
                  c_scale = 0, mode = "constant", targetClass = "minority"):
    log("training KNN")
    retVal = trainResult()

    # train
    ms_process_total = 0
    pTrain = ping()
    X_train_pre = pd.DataFrame(X_train).reset_index(drop=True).as_matrix()  # otherwise indices from X will be used
    X_test = pd.DataFrame(X_test).reset_index(drop=True).as_matrix().astype(float)  # otherwise indices from X will be used

    if (str(type(y_train)) != "<class 'numpy.ndarray'>"):
        y_train_pre = y_train.reset_index(drop=True)  # otherwise indices from X will be used
    else:
        y_train_pre = y_train

    if (str(type(y_test)) != "<class 'numpy.ndarray'>"):
        y_test = y_test.reset_index(drop=True).as_matrix()  # otherwise indices from X will be used
    from sklearn.model_selection import KFold

    bestAuROC = 0
    bestAuPRC = 0
    bestNN = 3
    neighborz = [3, 5, 10]

    number_of_folds = n_folds
    num_of_cpus = multiprocessing.cpu_count()

    for nn in neighborz:
        kf = KFold(n_splits=n_folds)
        kf.get_n_splits(X_train_pre)

        scoresAuROC = []
        scoresF1 = []
        scoresAccuracy = []
        scoresGmean = []
        scoresAuPRC = []

        for train_index, test_index in kf.split(X_train):
            X_train_cv, X_test_cv = X_train_pre[train_index], X_train_pre[test_index]
            y_train_cv, y_test_cv = y_train_pre[train_index], y_train_pre[test_index]

            # tests will be unaffected
            if (c_scale > 0):
                pProcess = ping()
                X_train_cv, y_train_cv = createScaledDataset(X_train_cv, y_train_cv, mode=mode, c=c_scale, targetClass=targetClass)
                ms_process_total += pong(pProcess)

            clf = neighbors.KNeighborsClassifier(n_neighbors = nn, n_jobs = num_of_cpus)

            # train
            if ((np.isnan(X_train_cv)).any):
                X_train_cv = np.nan_to_num(X_train_cv)

            clf.fit(X_train_cv, y_train_cv)
            pred = clf.predict(X_test_cv)

            # eval
            auROC, f1, accuracy, gmean, auPRC = getMetrics(y_test_cv, pred)

            scoresAuROC.append(auROC)
            scoresF1.append(f1)
            scoresAccuracy.append(accuracy)
            scoresGmean.append(gmean)
            scoresAuPRC.append(auPRC)

        meanScoreAuROC = np.mean(scoresAuROC)
        meanScoreF1 = np.mean(scoresF1)
        meanScoreAccuracy = np.mean(scoresAccuracy)
        meanScoreGmean = np.mean(scoresGmean)
        meanScoreAuPRC = np.mean(scoresAuPRC)

        if(meanScoreAuPRC > bestAuPRC):
            bestNN = nn
            bestAuROC = auROC
            bestAccuracy = meanScoreAccuracy
            bestGmean = meanScoreGmean
            bestF1 = meanScoreF1
            bestAuPRC = meanScoreAuPRC


    if (c_scale > 0):
        log("scaling final train data...")
        pProcess = ping()
        X_train, y_train = createScaledDataset(X_train, y_train,
                                               c=c_scale, mode=mode, targetClass=targetClass)
        ms_process_total /= len(neighborz)
        ms_process_total = pong(pProcess)

    if((np.isnan(X_train)).any):
        X_train= np.nan_to_num(X_train)
        
    clf = neighbors.KNeighborsClassifier(n_neighbors = bestNN, n_jobs = num_of_cpus)
    model = clf.fit(X_train, y_train)
    retVal.train_time = pong(pTrain)

    # get CV train metrics
    pTrainCV = ping()
    bestAccuracy, bestF1, bestGmean = -1, -1, -1
    retVal.train_eval_time = pong(pTrainCV)
    retVal.train_accuracy = bestAccuracy
    retVal.train_auc = bestAuROC
    retVal.train_f1 = bestF1
    retVal.train_gmean = bestGmean
    if(c_scale > 0):
        retVal.ms_process = ms_process_total

    pred = model.predict(X_test)

    lastModel = model
    lastY = y_test
    lastPred = pred

    pTest = ping()
    retVal.auROC, retVal.f1, retVal.accuracy, retVal.gmean, retVal.auPRC = getMetrics(y_test, pred)
    retVal.test_time = pong(pTest)

    log('KNN score: auc={}f, f1={}, accuracy={}, gmean={}'.format(retVal.auROC, retVal.f1, retVal.accuracy,
                                                                  retVal.gmean))

    return retVal

def trainRF(X_train, X_test, y_train, y_test):
    log("training RF")
    retVal = trainResult()

    
    # train    
    pTrain = ping()
    X_train_pre = pd.DataFrame(X_train).reset_index(drop=True).as_matrix() # otherwise indices from X will be used
    X_test = pd.DataFrame(X_test).reset_index(drop=True).as_matrix().astype(float) # otherwise indices from X will be used
            
            
    if (str(type(y_train)) != "<class 'numpy.ndarray'>"):
        y_train_pre = y_train.reset_index(drop=True) # otherwise indices from X will be used
    else:
         y_train_pre = y_train
            
    if (str(type(y_test)) != "<class 'numpy.ndarray'>"):
        y_test = y_test.reset_index(drop=True).as_matrix() # otherwise indices from X will be used
    from sklearn.model_selection import KFold
    bestAuROC = 0
    bestAuPRC = 0
    bestAccuracy = 0
    bestGmean = 0
    bestF1 = 0
    bestEstimators = 10
    bestCriterion = 'gini'
    estimatorz = [5,10,20]
    criterionz = ['gini', 'entropy']
    for estimators in estimatorz:
        for criterion in criterionz:
            kf = KFold(n_splits=n_folds)
            kf.get_n_splits(X_train_pre)
            
            scoresAuROC = []
            scoresF1 = []
            scoresAccuracy = []
            scoresGmean = []
            scoresAuPRC = []
                
            for train_index, test_index in kf.split(X_train):
                X_train_cv, X_test_cv = X_train_pre[train_index], X_train_pre[test_index]
                y_train_cv, y_test_cv = y_train_pre[train_index], y_train_pre[test_index]

                clf = RandomForestClassifier(n_estimators=estimators, criterion=criterion)
                
                # train
                clf.fit(X_train_cv, y_train_cv)
                pred = clf.predict(X_test_cv)

                # eval
                auROC, f1, accuracy, gmean, auPRC = getMetrics(y_test_cv, pred)
                
                scoresAuROC.append(auROC)
                scoresF1.append(f1)
                scoresAccuracy.append(accuracy)
                scoresGmean.append(gmean)
                scoresAuPRC.append(auPRC)

            meanScoreAuROC = np.mean(scoresAuROC)
            meanScoreF1 = np.mean(scoresF1)
            meanScoreAccuracy = np.mean(scoresAccuracy)
            meanScoreGmean = np.mean(scoresGmean)
            meanScoreAuPRC = np.mean(scoresAuPRC)

            if(meanScoreAuPRC > bestAuPRC):
                bestEstimators = estimators
                bestCriterion = criterion
                bestAuROC = auROC
                bestAccuracy = meanScoreAccuracy
                bestGmean = meanScoreGmean
                bestF1 = meanScoreF1
                bestAuPRC = meanScoreAuPRC
                
#     retVal.train_eval_time = 0
#     retVal.train_accuracy = bestAccuracy
#     retVal.train_auc = bestAuROC
#     retVal.train_f1 = bestF1
#     retVal.train_gmean = bestGmean
    
    clf = RandomForestClassifier(n_estimators=bestEstimators, criterion=bestCriterion)
    clf.fit(X_train_cv, y_train_cv)            
    retVal.train_time = pong(pTrain)
    
    # get CV train metrics
    pTrainCV = ping()
    bestAuROC = getCVPerformance(clf, X_train, y_train)
    bestAccuracy, bestF1, bestGmean = -1, -1, -1
    retVal.train_eval_time = pong(pTrainCV)
    retVal.train_accuracy = bestAccuracy
    retVal.train_auc = bestAuROC
    retVal.train_f1 = bestF1
    retVal.train_gmean = bestGmean
    
    pred = clf.predict(X_test)
    
    pTest = ping()
    retVal.auROC, retVal.f1, retVal.accuracy, retVal.gmean, retVal.auPRC = getMetrics(y_test, pred)
    retVal.test_time = pong(pTest)

    log('RF score: auROC={}f, auPRC={}f, f1={}, accuracy={}, gmean={}'.format(retVal.auROC, retVal.auPRC, retVal.f1, retVal.accuracy, retVal.gmean))
    
    return retVal



def trainRFScale(X_train, X_test, y_train, y_test, balanced = None,
                 c_scale = 0, mode = "constant", targetClass = "minority"):
    log("training RF")
    retVal = trainResult()
    ms_process_total = 0
    
    num_of_cpus = multiprocessing.cpu_count()

    
    # train    
    pTrain = ping()
    X_train_pre = pd.DataFrame(X_train).reset_index(drop=True).as_matrix() # otherwise indices from X will be used
    X_test = pd.DataFrame(X_test).reset_index(drop=True).as_matrix().astype(float) # otherwise indices from X will be used
            
            
    if (str(type(y_train)) != "<class 'numpy.ndarray'>"):
        y_train_pre = y_train.reset_index(drop=True) # otherwise indices from X will be used
    else:
         y_train_pre = y_train
            
    if (str(type(y_test)) != "<class 'numpy.ndarray'>"):
        y_test = y_test.reset_index(drop=True).as_matrix() # otherwise indices from X will be used
    from sklearn.model_selection import KFold
    bestAuROC = 0
    bestAuPRC = 0
    bestAccuracy = 0
    bestGmean = 0
    bestF1 = 0
    bestEstimators = 10
    bestCriterion = 'gini'
    estimatorz = [5,10,20]
    criterionz = ['gini', 'entropy']
    for estimators in estimatorz:
        for criterion in criterionz:
            kf = KFold(n_splits=n_folds)
            kf.get_n_splits(X_train_pre)
            
            scoresAuROC = []
            scoresF1 = []
            scoresAccuracy = []
            scoresGmean = []
            scoresAuPRC = []
                
            for train_index, test_index in kf.split(X_train):
                X_train_cv, X_test_cv = X_train_pre[train_index], X_train_pre[test_index]
                y_train_cv, y_test_cv = y_train_pre[train_index], y_train_pre[test_index]

                clf = RandomForestClassifier(n_estimators=estimators, criterion=criterion, 
                                             class_weight = balanced, n_jobs = num_of_cpus)
                
                # tests will be unaffected
                if (c_scale > 0):
                    pProcess = ping()
                    X_train_cv, y_train_cv = createScaledDataset(X_train_cv, y_train_cv, mode=mode, c=c_scale, targetClass=targetClass)
                    ms_process_total += pong(pProcess)                    

                if ((np.isnan(X_train_cv)).any):
                    X_train_cv = np.nan_to_num(X_train_cv)
                    
                # train
                clf.fit(X_train_cv, y_train_cv)
                pred = clf.predict(X_test_cv)

                # eval
                auROC, f1, accuracy, gmean, auPRC = getMetrics(y_test_cv, pred)
                
                scoresAuROC.append(auROC)
                scoresF1.append(f1)
                scoresAccuracy.append(accuracy)
                scoresGmean.append(gmean)
                scoresAuPRC.append(auPRC)

            meanScoreAuROC = np.mean(scoresAuROC)
            meanScoreF1 = np.mean(scoresF1)
            meanScoreAccuracy = np.mean(scoresAccuracy)
            meanScoreGmean = np.mean(scoresGmean)
            meanScoreAuPRC = np.mean(scoresAuPRC)

            if(meanScoreAuPRC > bestAuPRC):
                bestEstimators = estimators
                bestCriterion = criterion
                bestAuROC = auROC
                bestAccuracy = meanScoreAccuracy
                bestGmean = meanScoreGmean
                bestF1 = meanScoreF1
                bestAuPRC = meanScoreAuPRC
                
    if (c_scale > 0):
        log("scaling final train data...")
        pProcess = ping()
        X_train, y_train = createScaledDataset(X_train, y_train,
                                               c=c_scale, mode=mode, targetClass=targetClass)
        ms_process_total /= len(estimatorz)
        ms_process_total /= len(criterionz)
        ms_process_total = pong(pProcess)
        
    if((np.isnan(X_train)).any):
        X_train= np.nan_to_num(X_train)
        
    clf = RandomForestClassifier(n_estimators=bestEstimators, criterion=bestCriterion)
    clf.fit(X_train_cv, y_train_cv)            
    retVal.train_time = pong(pTrain)
    
    # get CV train metrics
    pTrainCV = ping()
    bestAuROC = bestAuROC
    bestAccuracy, bestF1, bestGmean = -1, -1, -1
    retVal.train_eval_time = pong(pTrainCV)
    retVal.train_accuracy = bestAccuracy
    retVal.train_auc = bestAuROC
    retVal.train_f1 = bestF1
    retVal.train_gmean = bestGmean
    if(c_scale > 0):
        retVal.ms_process = ms_process_total
    
    pred = clf.predict(X_test)
    
    pTest = ping()
    retVal.auROC, retVal.f1, retVal.accuracy, retVal.gmean, retVal.auPRC = getMetrics(y_test, pred)
    retVal.test_time = pong(pTest)

    log('RF score: auROC={}f, auPRC={}f, f1={}, accuracy={}, gmean={}'.format(retVal.auROC, retVal.auPRC, retVal.f1, retVal.accuracy, retVal.gmean))
    
    return retVal



def trainLR(X_train, X_test, y_train, y_test, balanced = None, scoring = "none"):
    """
    Trains a logistic regression based on cross validation.
    :param balance: Use 'balanced' or None
        Weights associated with classes in the form {class_label: weight}. 
        If not given, all classes are supposed to have weight one.
    
        The “balanced” mode uses the values of y to automatically adjust weights inversely 
        proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
        
        Note that these weights will be multiplied with sample_weight (passed through the 
        fit method) if sample_weight is specified. 
        From http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
    :param scoring: Use "none" for default, "auROC" for AUC of ROC curve
    """
    log("training LR ({})".format(scoring))
    
    retVal = trainResult()
    
    number_of_folds = n_folds
    tolerance = 0.01
    num_of_cpus = multiprocessing.cpu_count()
    
    
    if (scoring == "auROCWeighted"):
        log("\"auROCWeighted\" set. Using area under ROC curve with weighted samples...")
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))
        w = n_samples / (n_classes * np.bincount(y_train)) # bincount returns the number of instances for non-negative integer: 0, 1, ...
        # w holds now the inverse weights of all classes
        w_array = w[y_train] # pick weight based on corresponding label
        scorer = auc_scorer = make_scorer(roc_auc_score,
                                          average = "weighted",
                                          sample_weight = w_array) # additional parameters can be specified, see 
                                            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
                                            # and http://scikit-learn.org/dev/modules/model_evaluation.html
    elif (scoring == "auROC"):
        log("\"auROC\" set. Using area under ROC curve...")
        scorer = auc_scorer = make_scorer(roc_auc_score) # additional parameters can be specified, see 
                                            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
                                            # and http://scikit-learn.org/dev/modules/model_evaluation.html
    else:
        log("Scoring method \"" + scoring + "\" not recognized or set. Using default (accuracy)...")
        scorer = None
    
                
    lr = linear_model.LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], 
                                          class_weight = balanced,
                                          cv = number_of_folds,
                                          penalty = 'l1',
                                          scoring = scorer,
                                          solver = 'liblinear',
                                          tol = tolerance,
                                          n_jobs = num_of_cpus)
        
        

    tain_time = 0
    test_time = 0
    
    
    pTrain = ping()
    model = lr.fit(X_train, y_train)
    retVal.train_time = pong(pTrain)
        
    # for logisticRegressionCV the cv is already built in, therefore, we can use clf.scores_[1]:
    # clf.scores_[1].shape > (6, 3) > 6 = number of folds, 3 = number of tried out Cs.
    pTrainCV = ping()
    #bestAuROC = getCVPerformance(model, X_train, y_train)
    aucMeanCspecific = max(np.mean(model.scores_[1], axis = 0)) # get CV train metrics
    #bestAuROC = max(np.mean(model.scores_[1], axis = 0)) # Take the mean for each C and then the maximum
    bestAccuracy, bestF1, bestGmean = -1, -1, -1
    retVal.train_eval_time = pong(pTrainCV)
    retVal.train_accuracy = bestAccuracy
    retVal.train_auc = aucMeanCspecific
    retVal.train_f1 = bestF1
    retVal.train_gmean = bestGmean
    log("best AUC during training was" + str(aucMeanCspecific))
    
    pred_all = model.predict_proba(X_test)
    pred = pred_all[:,1]
    
    #pprint(np.column_stack((y_test, pred.round(3))))
    
    pTest = ping()
    retVal.auROC, retVal.f1, retVal.accuracy, retVal.gmean, retVal.auPRC = getMetrics(y_test, pred)
    retVal.test_time = pong(pTest)
    
    log('LR score: auROC={}f, auPRC={}f, f1={}, accuracy={}, gmean={}'.format(retVal.auROC, retVal.auPRC, retVal.f1, retVal.accuracy, retVal.gmean))
    
    return retVal

def trainLRScale(X_train, X_test, y_train, y_test, balanced = None,
                 c_scale = 0, mode = "constant", targetClass = "minority"):
    """
    Trains a logistic regression based on cross validation.
    :param balance: Use 'balanced' or None
        Weights associated with classes in the form {class_label: weight}. 
        If not given, all classes are supposed to have weight one.
    
        The “balanced” mode uses the values of y to automatically adjust weights inversely 
        proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
        
        Note that these weights will be multiplied with sample_weight (passed through the 
        fit method) if sample_weight is specified. 
        From http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
    :param scoring: Use "none" for default, "auROC" for AUC of ROC curve
    """
    
    retVal = trainResult()
    
    number_of_folds = n_folds
    tolerance = 0.01
    num_of_cpus = multiprocessing.cpu_count()
    ms_process_total = 0
    
    # train    
    pTrain = ping()
    X_train_pre = pd.DataFrame(X_train).reset_index(drop=True).as_matrix() # otherwise indices from X will be used
    X_test = pd.DataFrame(X_test).reset_index(drop=True).as_matrix().astype(float) # otherwise indices from X will be used
            
            
    if (str(type(y_train)) != "<class 'numpy.ndarray'>"):
        y_train_pre = y_train.reset_index(drop=True) # otherwise indices from X will be used
    else:
        y_train_pre = y_train
            
    if (str(type(y_test)) != "<class 'numpy.ndarray'>"):
        y_test = y_test.reset_index(drop=True).as_matrix() # otherwise indices from X will be used
    from sklearn.model_selection import KFold
    bestAuROC = 0
    bestAuPRC = 0
    bestC= 0.001
    Cs = [0.001, 0.01, 0.1, 1]
    for c in Cs:
        kf = KFold(n_splits=n_folds)
        kf.get_n_splits(X_train_pre)

        scoresAuROC = []
        scoresF1 = []
        scoresAccuracy = []
        scoresGmean = []
        scoresAuPRC = []
        
        for train_index, test_index in kf.split(X_train):
            X_train_cv, X_test_cv = X_train_pre[train_index], X_train_pre[test_index]
            y_train_cv, y_test_cv = y_train_pre[train_index], y_train_pre[test_index]
            
            # tests will be unaffected
            if (c_scale > 0):
                pProcess = ping()
                X_train_cv, y_train_cv = createScaledDataset(X_train_cv, y_train_cv, mode = mode, c = c_scale, targetClass = targetClass)
                ms_process_total += pong(pProcess)

            clf = linear_model.LogisticRegression(C=c, 
                                          class_weight = balanced,
                                          penalty = 'l1',
                                          solver = 'liblinear',
                                          tol = tolerance,
                                          n_jobs = num_of_cpus)

            # train
            if((np.isnan(X_train_cv)).any):
                X_train_cv= np.nan_to_num(X_train_cv)
                
            clf.fit(X_train_cv, y_train_cv)
            pred = clf.predict(X_test_cv)

            # eval
            auROC, f1, accuracy, gmean, auPRC = getMetrics(y_test_cv, pred)

            scoresAuROC.append(auROC)
            scoresF1.append(f1)
            scoresAccuracy.append(accuracy)
            scoresGmean.append(gmean)
            scoresAuPRC.append(auPRC)

        meanScoreAuROC = np.mean(scoresAuROC)
        meanScoreF1 = np.mean(scoresF1)
        meanScoreAccuracy = np.mean(scoresAccuracy)
        meanScoreGmean = np.mean(scoresGmean)
        meanScoreAuPRC = np.mean(scoresAuPRC)

        if(meanScoreAuPRC > bestAuPRC):
            bestC = c
            bestAuROC = auROC
            bestAccuracy = meanScoreAccuracy
            bestGmean = meanScoreGmean
            bestF1 = meanScoreF1
            bestAuPRC = meanScoreAuPRC
    
    tain_time = 0
    test_time = 0
    lr = linear_model.LogisticRegression(C=bestC, 
                                          class_weight = balanced,
                                          penalty = 'l1',
                                          solver = 'liblinear',
                                          tol = tolerance,
                                          n_jobs = num_of_cpus)
    if (c_scale > 0): 
        log("scaling final train data...")
        pProcess = ping()
        X_train, y_train = createScaledDataset(X_train, y_train,
                                                     c = c_scale, mode = mode, targetClass = targetClass)
        ms_process_total /= len(Cs)
        ms_process_total = pong(pProcess)

    if((np.isnan(X_train)).any):
        X_train= np.nan_to_num(X_train)
    model = lr.fit(X_train, y_train)
    retVal.train_time = pong(pTrain)
    
    pTrainCV = ping()
    aucMeanCspecific = bestAuROC
    bestAccuracy, bestF1, bestGmean = -1, -1, -1
    retVal.train_eval_time = pong(pTrainCV)
    retVal.train_accuracy = bestAccuracy
    retVal.train_auc = aucMeanCspecific
    retVal.train_f1 = bestF1
    retVal.train_gmean = bestGmean
    log("best AUC during training was" + str(aucMeanCspecific))
    if(c_scale > 0):
        retVal.ms_process = ms_process_total
    
    pred_all = model.predict_proba(X_test)
    pred = pred_all[:,1]
    
    #pprint(np.column_stack((y_test, pred.round(3))))
    
    pTest = ping()
    retVal.auROC, retVal.f1, retVal.accuracy, retVal.gmean, retVal.auPRC = getMetrics(y_test, pred)
    retVal.test_time = pong(pTest)
    
    log('LR score: auROC={}f, auPRC={}f, f1={}, accuracy={}, gmean={}'.format(retVal.auROC, retVal.auPRC, retVal.f1, retVal.accuracy, retVal.gmean))
    
    return retVal


# ## Main loop, where all targets will be called subsequently

# In[ ]:

def evalAll(X_train_pre, X_test_pre, y_train_pre, y_test_pre, dataset, target, bs2_measure,
            modeltypes = ['RF', 'OCC', 'LR', 'KNN', 'WLR'], 
            approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ScaleMajority', 'ScaleMinority', 'ScaleBoth', 'ADASYN']):
    global X_train, X_test, y_train, y_test
    
    nAllSample = len(X_train_pre) + len(X_test_pre)
    nAllSamplePos = sum(y_train_pre) + sum(y_test_pre)
    nAllSampleNeg = nAllSample - nAllSamplePos
    
    nTrainSampleBefore = len(X_train_pre)
    nTrainSamplePosBefore = sum(y_train_pre)
    nTrainSampleNegBefore = nTrainSampleBefore - nTrainSamplePosBefore
                
    for approach in approaches:
        c_opt_cv = -1
        if target == "BEFUND___TA___61_14_535":
            c_manual_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            c_manual_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            
        for c_manual in c_manual_values:
            for scalingMode in ["linear", "constant"]:
                msProcess = 0 # reset
                approachMod = approach
                if("Scale" in approach):
                    log("trying approach " + approach + ", using c=" + str(c_manual) + "...")
                elif(c_manual == 0.0 and scalingMode == "constant"):
                    log("trying approach " + approach + "...")
                    c_opt_cv = -1
                else:
                    continue # only scaling makes sense to be analysed with various c's

                approachMod = approach
                if('naive' in approach):
                    pProcess = ping()
                    X_test, y_test = X_test_pre, y_test_pre
                    X_train, y_train = createNaiveDataset(X_train_pre, y_train_pre)
                    msProcess = pong(pProcess)

                if('SMOTE' in approach):
                    pProcess = ping()
                    X_test, y_test = X_test_pre, y_test_pre
                    X_train, y_train = createSMOTEDataset(X_train_pre, y_train_pre)
                    msProcess = pong(pProcess)

                if('tomek' in approach):
                    pProcess = ping()
                    X_test, y_test = X_test_pre, y_test_pre
                    X_train, y_train = createTomekDataset(X_train_pre, y_train_pre)                
                    msProcess = pong(pProcess)

                if('ADASYN' in approach):
                    pProcess = ping()
                    X_test, y_test = X_test_pre, y_test_pre
                    X_train, y_train = createADASYNDataset(X_train_pre, y_train_pre)     
                    msProcess = pong(pProcess)

                if('OSS' in approach):
                    pProcess = ping()
                    X_test, y_test = X_test_pre, y_test_pre
                    X_train, y_train = createOSSDataset(X_train_pre, y_train_pre)  
                    msProcess = pong(pProcess)

                if('CNN' in approach):
                    pProcess = ping()
                    X_test, y_test = X_test_pre, y_test_pre
                    X_train, y_train = createCNNDataset(X_train_pre, y_train_pre)  
                    msProcess = pong(pProcess)

                targetClass = ""
                if('ScaleMajority' in approach):
                    #pProcess = ping()
                    targetClass = "majority"
                    approachMod = "CSS"
                    X_test, y_test = X_test_pre, y_test_pre
                    X_train, y_train = X_train_pre, y_train_pre
                    #X_train, y_train = createScaledDataset(X_train_pre, y_train_pre,c = c_manual, mode = scalingMode,
                    #                                       targetClass = targetClass 
                    #                                       )
                    #msProcess = pong(pProcess)      

                if('ScaleMinority' in approach):
                    #pProcess = ping()
                    targetClass = "minority"
                    approachMod = "CSS"
                    X_test, y_test = X_test_pre, y_test_pre
                    X_train, y_train = X_train_pre, y_train_pre
                    #X_train, y_train = createScaledDataset(X_train_pre, y_train_pre, 
                    #                                       c = c_manual, mode = scalingMode,
                    #                                       targetClass = targetClass)
                    #msProcess = pong(pProcess)

                if('ScaleBoth' in approach):
                    #pProcess = ping()
                    targetClass = "both"
                    approachMod = "CSS"
                    X_test, y_test = X_test_pre, y_test_pre
                    X_train, y_train = X_train_pre, y_train_pre
                    #X_train, y_train = createScaledDataset(X_train_pre, y_train_pre, 
                    #                                       c = c_manual, mode = scalingMode,
                    #                                       targetClass = targetClass)
                    #msProcess = pong(pProcess)

                for modeltype in modeltypes:
                    log("evaluating approach {} using model {}...".format(approach, modeltype))
                    
                    #try:
                    if(modeltype == 'LR'):
                        log("debug: {}, {}, {}".format(c_manual, scalingMode, targetClass))
                        retVal = trainLRScale(X_train, X_test, y_train, y_test, balanced=None,
                                              c_scale = c_manual, mode = scalingMode, targetClass = targetClass)

                    if(modeltype == 'KNN'):
                        retVal = trainKNNScale(X_train, X_test, y_train, y_test,
                                              c_scale = c_manual, mode = scalingMode, targetClass = targetClass)

                    if(modeltype == 'WLR'):
                        retVal = trainLRScale(X_train, X_test, y_train, y_test, balanced='balanced',
                                              c_scale = c_manual, mode = scalingMode, targetClass = targetClass)

                    if(modeltype == 'OCC'):
                        retVal = trainOCCScale(X_train, X_test, y_train, y_test,
                                              c_scale = c_manual, mode = scalingMode, targetClass = targetClass)

                    if(modeltype == 'RF'):
                        retVal = trainRFScale(X_train, X_test, y_train, y_test, 
                                              c_scale = c_manual, mode = scalingMode, targetClass = targetClass)

                    if(modeltype == 'WRF'):
                        retVal = trainRFScale(X_train, X_test, y_train, y_test, balanced='balanced',
                                              c_scale = c_manual, mode = scalingMode, targetClass = targetClass)
                        
                    if(modeltype == 'NN'):
                        retVal = trainNNScale(X_train, X_test, y_train, y_test,
                                              c_scale = c_manual, mode = scalingMode, targetClass = targetClass)
                    #except Exception as e:
                    #    log("building model failed:" + str(e))

                    dummy, nFeature = X_train.shape
                    nTrainSample = len(X_train)
                    nTrainSamplePos = sum(y_train)
                    nTrainSampleNeg = nTrainSample - nTrainSamplePos

                    if(retVal.ms_process != -1):
                        log("ret val != -1. Using ms process from model ({}).".format(retVal.ms_process))
                        msProcess = retVal.ms_process
                    else:
                        log("preprocessing took {}ms".format(msProcess))

                    Report.logToFile(target = target, dataset = dataset, model_type=modeltype, 
                                     model_train_time=retVal.train_time, 
                                     model_train_eval_time=retVal.train_eval_time, 
                                     model_test_time=retVal.test_time,
                                     model_accuracy= retVal.accuracy, 
                                     model_auroc = retVal.auROC, 
                                     model_auprc = retVal.auPRC, 
                                     model_f1 = retVal.f1, 
                                     model_gmean = retVal.gmean, 
                                     model_train_accuracy= retVal.train_accuracy,
                                     model_train_auroc = retVal.train_auroc,
                                     model_train_auprc = retVal.train_auprc,
                                     model_train_f1 = retVal.train_f1, 
                                     model_train_gmean = retVal.train_gmean, 
                                     num_features = nFeature, 
                                     num_sample_dataset = nAllSample,
                                     num_sample_dataset_pos = nAllSamplePos,
                                     num_sample_dataset_neg = nAllSampleNeg,
                                     num_sample_train_before = nTrainSampleBefore,
                                     num_sample_train_before_pos = nTrainSamplePosBefore,
                                     num_sample_train_before_meg = nTrainSampleNegBefore,
                                     num_sample_train_after = nTrainSample,
                                     num_sample_train_after_pos = nTrainSamplePos, 
                                     num_sample_train_after_neg = nTrainSampleNeg,
                                     bs2 = bs2_measure,
                                     process_time = msProcess, 
                                     process_name = approachMod,
                                     process_naive = 1 if ('naive' in approach) else 0, 
                                     process_sampling_up_smote = 1 if ('SMOTE' in approach) else 0,  
                                     process_sampling_up_adasyn = 1 if ('ADASYN' in approach) else 0, 
                                     process_sampling_down_oss = 1 if ('OSS' in approach) else 0, 
                                     process_sampling_down_cnn = 1 if ('CNN' in approach) else 0,  
                                     process_sampling_down_tomek = 1 if ('tomek' in approach) else 0, 
                                     process_weight = 1 if (modeltype == 'WLR') else 0, 
                                     process_scale_minority = 1 if ('CSS' in approachMod) else 0, 
                                     process_scale_mode = scalingMode if ('CSS' in approachMod) else "", 
                                     process_scale_target = targetClass if ('CSS' in approachMod) else "", 
                                     process_scale_c = c_manual if ('CSS' in approachMod) else 0)
                    
                    # notify ubidots
                    if use_ubi:
                        try:
                            new_value = ubi_last_timestamp.save_value({'value': 10, 'context':{'lastTimestamp': "'" + str(datetime.datetime.now()) + "'"}})
                        except Exception as e:
                            log("ubidots failed." + str(e))
                    


# In[ ]:

def evalSubSC(dfIn, name, classColumnName, modeltypes, approaches,
             indexFeatureStart, indexFeatureEnd, fixedSplit = "no"):
    X, y = createDatasetXY(df = dfIn, indexFeatureStart = indexFeatureStart,
                                       indexFeatureEnd= indexFeatureEnd, targetColumnName=classColumnName)    
    bs2_measure = getBS2(X = X, y = y)

    nSample, nFeature = X.shape
    nSampleNeg = nSample - sum(y)
    
    if (fixedSplit == "no"):
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split_scaled(X, y) 
    else:
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split_scaled(X, y, dfIn[fixedSplit]) 
        
        
    evalAll(X_train_pre, X_test_pre, y_train_pre, y_test_pre, dataset = name, target = name, bs2_measure = bs2_measure,
               modeltypes = modeltypes, approaches = approaches)

    log("completed " + name)


# In[ ]:

def evalSubMC(dfIn, name, classColumnName, indices, modeltypes, approaches,
             indexFeatureStart, indexFeatureEnd, fixedSplit = "no"):
    classColumnName 
    for i in indices:
        log("working " + name + " " + str(i))
        dfSub = dfIn.copy()
        dfSub[classColumnName] = (dfSub[classColumnName] == i).astype(bool)
        dfSub[classColumnName] = dfSub[classColumnName].astype(int)

        X, y = createDatasetXY(df = dfSub, indexFeatureStart = indexFeatureStart, 
                                         indexFeatureEnd= indexFeatureEnd, targetColumnName=classColumnName)  
        bs2_measure = getBS2(X = X, y = y)
        
        nSample, nFeature = X.shape
        nSampleNeg = len(y) - sum(y)
        log("{} samples total, {} negative and {} positive".format(nSample, nSampleNeg, (nSample-nSampleNeg)))
        if(fixedSplit == "no"):
            X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split_scaled(X, y) 
        else:
            X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split_index(X, y, dfSub[fixedSplit])
        evalAll(X_train_pre, X_test_pre, y_train_pre, y_test_pre, dataset = name, target = name + str(i), 
                bs2_measure = bs2_measure, modeltypes = modeltypes, approaches = approaches)

    log("completed " + name)


# In[ ]:

def evalSubGlas(modeltypes = ['RF', 'WRF', 'OCC', 'LR', 'KNN', 'WLR'], 
                approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 'ScaleMajority', 'ScaleMinority', 'ScaleBoth']):
    log("starting glas", force=True)
    evalSubMC(dfGlass, "Glass", "Type", [1,2,3,5,6,7], modeltypes, approaches, 0, ncGlass-1)


# In[ ]:

def evalSubGlas67(modeltypes = ['RF', 'WRF',  'OCC', 'LR', 'KNN', 'WLR'], 
                approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 'ScaleMajority', 'ScaleMinority', 'ScaleBoth']):
    evalSubMC(dfGlass, "Glass", "Type", [6,7], modeltypes, approaches, 0, ncGlass-1)


# In[ ]:

def evalSubVowel(modeltypes = ['RF', 'WRF',  'OCC', 'LR', 'KNN', 'WLR'], 
                 approaches = [ 'OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 'ScaleMajority', 'ScaleMinority', 'ScaleBoth']):
    log("starting Vowel", force=True)
    evalSubMC(dfVowel, "Vowel", "Class", range(0,11), modeltypes, approaches, 1, ncVowel - 2, fixedSplit = "Train or Test")


# In[ ]:

def evalSubForest(modeltypes = ['RF', 'WRF', 'OCC', 'LR', 'KNN', 'WLR'], 
                 approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 'ScaleMajority', 'ScaleMinority', 'ScaleBoth']):
    log("starting forest", force=True)
    evalSubSC(dfForest, "Forest", "area", modeltypes, approaches, 0, ncForest-1)


# In[ ]:

def evalSubPima(modeltypes = ['RF', 'WRF', 'OCC', 'LR', 'KNN', 'WLR'], 
                 approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 
                               'ScaleMajority', 'ScaleMinority', 'ScaleBoth']):
    log("starting pima", force=True)
    evalSubSC(dfPima, "Pima", "Class", modeltypes, approaches, 0, 8)


# In[ ]:

def evalSubPhoneme(modeltypes = ['RF', 'WRF', 'OCC', 'LR', 'KNN', 'WLR'], 
                 approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 
                               'ScaleMajority', 'ScaleMinority', 'ScaleBoth']):
    log("starting phoneme", force=True)
    evalSubSC(dfPhoneme, "Phoneme", "class", modeltypes, approaches, 0, 5)


# In[ ]:

def evalSubVehicle(modeltypes = ['RF', 'WRF', 'OCC', 'LR', 'KNN', 'WLR'], 
                 approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 
                               'ScaleMajority', 'ScaleMinority', 'ScaleBoth']):
    log("starting vehicle", force=True)
    evalSubMC(dfVehicle, "Vehicle", "TARGET", [1, 2, 3, 4], modeltypes, approaches, 0, 18)


# In[ ]:

def evalSubAbalone(modeltypes = ['RF', 'WRF', 'OCC', 'LR', 'KNN', 'WLR'], 
                 approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 
                               'ScaleMajority', 'ScaleMinority', 'ScaleBoth']):
    log("starting abalone", force=True)
    evalSubSC(dfAbalone, "Abalone", "Rings", modeltypes, approaches, 0, 8)


# In[ ]:

def evalSubSatimage(modeltypes = ['RF', 'WRF', 'OCC', 'LR', 'KNN', 'WLR'], 
                 approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 
                               'ScaleMajority', 'ScaleMinority', 'ScaleBoth']):
    log("starting satimage", force=True)
    evalSubMC(dfSatimage, "Satimage", "CLASS", [1,2,3,4,5,7], modeltypes, approaches, 0, 18, fixedSplit = "TRAIN_TEST")


# In[ ]:

def evalSubMammography(modeltypes = ['RF', 'WRF', 'OCC', 'LR', 'KNN', 'WLR'], 
                 approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 
                               'ScaleMajority', 'ScaleMinority', 'ScaleBoth']):
    log("starting mammography", force=True)
    evalSubSC(dfMammography, "Mammography", "target", modeltypes, approaches, 0, 6)


# In[ ]:

def evalSubAutomotive(modeltypes = ['LR', 'KNN', 'WLR', 'RF', 'WRF', 'OCC'],
                      approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 'ScaleMajority', 'ScaleMinority', 'ScaleBoth'],
                      strategies = ['Random', 'Planned']):
    # read in targets
    targets = pd.DataFrame.from_csv("data/targets_w_lambda_sub_sub.csv").sort_values(by = ['CNT'])
    rows, cols = targets.shape
    i = 1

    # eval each
    for target in targets['TARGET']:
        #target = "DTC___1196802" # 0x 124302 = 1196802 is the lambda rex thing, 1250
        #target = "DTC___1257473" # 21 instances, first random entry
        log("starting automotive (target= {})".format(target), force=True)

        # determine prefixes to drop (aside from prefix of target, which will be dropped automatically)
        prefixesToDrop = []
        if (target.startswith("DTC___")):
            prefixesToDrop = ["BEFUND___", "DK___"]
        elif (target.startswith("BEFUND___")):
            prefixesToDrop = ["DK___"]


        dfTemp = getDataFrameForTarget(dfAutomotive.copy(), target, prefixesToDrop)
        # move "TARGET" to start
        target_col = dfTemp['TARGET']
        dfTemp.drop(labels=['TARGET'], axis=1,inplace = True)
        dfTemp.insert(0, 'TARGET', target_col)

        for strat in strategies:
            XAutomotive, yAutomotive = 0, 0
            if strat == 'Planned':
                XAutomotive, yAutomotive = createDatasetUsingMetaXy(df = dfTemp.copy(), indexFeatureStart = 1,
                                                                    indexFeatureEnd= ncAutomotive, targetColumnName="TARGET",
                                                                    metaColumnName = "META___PLANNED")

            if strat == 'Random':
                XAutomotive, yAutomotive = createDatasetUsingRandomXy(df = dfTemp.copy(), indexFeatureStart = 1,
                                                                    indexFeatureEnd= ncAutomotive, ratio = 100)
                
               

            XAutomotive = dropPrefix(XAutomotive, "META___")

            # do PCA in any case
            XAutomotive = doPCA(XAutomotive, numberOfDimensionsTarget = 100)
            bs2_measure = getBS2(X =XAutomotive, y = yAutomotive)

            
            X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split_scaled(XAutomotive, yAutomotive) 
            
            log("types: X_train_pre={}, X_test_pre={}, y_train_pre={}, y_test_pre={}".format(str(type(X_train_pre)),
                                                                                             str(type(X_test_pre)),
                                                                                             str(type(y_train_pre)),
                                                                                             str(type(y_test_pre))))

            nRowsBasic, nColsBasic = X_train_pre.shape    
            log("basic shape of train dataset is {} rows and {} features.".format(nRowsBasic, nColsBasic))

            evalAll(X_train_pre, X_test_pre, y_train_pre, y_test_pre, dataset = "Automotive_" + strat , target = target,
                    bs2_measure = bs2_measure, modeltypes = modeltypes, approaches = approaches)
    log("completed automotive")


# In[ ]:

def doIt_stepwise():
    global logpath
    smoothing = 5
    base_log_path = "log/"
    
    modeltypes = ['RF', 'WRF', 'OCC', 'LR', 'KNN', 'WLR'] # no NN, painfully slow
    approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 'ScaleMajority', 'ScaleMinority', 'ScaleBoth']

    for i in range(0,smoothing):
        logpath = base_log_path + str(i) + "_" + "mammography" + ".csv"
        evalSubMammography(approaches = approaches, modeltypes = modeltypes)
        
    for i in range(0,smoothing):
        logpath = base_log_path + str(i) + "_" + "satimage" + ".csv"
        evalSubSatimage(approaches = approaches, modeltypes = modeltypes)
        
    for i in range(0,smoothing):
        logpath = base_log_path + str(i) + "_" + "vowel" + ".csv"
        evalSubVowel(approaches = approaches, modeltypes = modeltypes)
        
    for i in range(0,smoothing):
        logpath = base_log_path + str(i) + "_" + "forest" + ".csv"
        evalSubForest(approaches = approaches, modeltypes = modeltypes)
        
    for i in range(0,smoothing):
        logpath = base_log_path + str(i) + "_" + "glass" + ".csv"
        evalSubGlas(approaches = approaches, modeltypes = modeltypes)
        
    for i in range(0,smoothing):
        logpath = base_log_path + str(i) + "_" + "pima" + ".csv"
        evalSubPima(approaches = approaches, modeltypes = modeltypes)
        
    for i in range(0,smoothing):
        logpath = base_log_path + str(i) + "_" + "phoneme" + ".csv"
        evalSubPhoneme(approaches = approaches, modeltypes = modeltypes)
        
    for i in range(0,smoothing):
        logpath = base_log_path + str(i) + "_" + "vehicle" + ".csv"
        evalSubVehicle(approaches = approaches, modeltypes = modeltypes)
        
    for i in range(0,smoothing):
        logpath = base_log_path + str(i) + "_" + "abalone" + ".csv"
        evalSubAbalone(approaches = approaches, modeltypes = modeltypes)
        
    for i in range(0,smoothing):
        logpath = base_log_path + str(i) + "_" + "automotive" + ".csv"
        evalSubAutomotive(approaches = approaches, modeltypes = modeltypes)
        
doIt_stepwise()


# In[ ]:

def doIt():
    smoothing = 5

    for i in range(0,smoothing):
        log("starting smoothing iteration {}".format(i), force=True)

        #approaches = ['ScaleBoth']
        #modeltypes = ['OCC']
        modeltypes = ['RF', 'WRF', 'OCC', 'LR', 'KNN', 'WLR', 'NN']
        approaches = ['OSS', 'CNN', 'naive', 'SMOTE', 'tomek', 'ADASYN', 'ScaleMajority', 'ScaleMinority', 'ScaleBoth']

        evalSubMammography(approaches = approaches, modeltypes = modeltypes)
        evalSubSatimage(approaches = approaches, modeltypes = modeltypes)
        evalSubVowel(approaches = approaches, modeltypes = modeltypes)
        evalSubForest(approaches = approaches, modeltypes = modeltypes)
        evalSubGlas(approaches = approaches, modeltypes = modeltypes)
        evalSubPima(approaches = approaches, modeltypes = modeltypes)
        evalSubPhoneme(approaches = approaches, modeltypes = modeltypes)
        evalSubVehicle(approaches = approaches, modeltypes = modeltypes)
        evalSubAbalone(approaches = approaches, modeltypes = modeltypes)
        evalSubAutomotive(strategies = ['Random'], approaches = approaches, modeltypes = modeltypes)

#doIt()


# In[ ]:

if False:
    def doIt():
        smoothing = 2

        for i in range(0,smoothing):
            log("starting smoothing iteration {}".format(i))

            modeltypes = ['NN']

            evalSubMammography(modeltypes = modeltypes)
            evalSubSatimage(modeltypes = modeltypes)
            evalSubVowel(modeltypes = modeltypes)
            evalSubForest(modeltypes = modeltypes)        
            evalSubGlas(modeltypes = modeltypes)
            evalSubPima(modeltypes = modeltypes)
            evalSubPhoneme(modeltypes = modeltypes)
            evalSubVehicle(modeltypes = modeltypes)
            evalSubAbalone(modeltypes = modeltypes)
            evalSubAutomotive(strategies = ['Random'], modeltypes = modeltypes)

    doIt()


# In[ ]:

if False:
    evalSubAutomotive(modeltypes = ['LR', 'KNN', 'WLR', 'RF'],
                      strategies = ['Planned'])
    evalSubAutomotive(modeltypes = ['OCC'],
                      strategies = ['Planned'])


# The following results are stored to CSV:
# - TARGET: Name of the target, e.g. "DTC\_\_\_12345" or "Vowel1" or "class0"
# - DATASET: Dataset, e.g. "automotive"
# - MODEL_TYPE = "KNN", "LR" or "OCC" (one class classifier)
# - MODEL_TRAIN_TIME: Training time of the Knn in ms
# - MODEL_ACCURACY: Accuracy achieved by the classifier
# - MODEL_AUC: AUC achieved by the classifier
# - MODEL_F1: F1 achieved by the classifier
# - MODEL_GMEAN: the g performance measure
# - NUM_FEATURES: The number of features (after being compressed by PCA in the automotive set)
# - NUM_SAMPLES_POS: The number of observations where (minority) target class was present
# - NUM_SAMPLES_NEG: The number of negative observations, where target class was not present
# - PROCESS_TIME: time in ms it took to complete all processing steps
# - PROCESS_NAIVE: 0/1 if dataset has been processed the naive way
# - PROCESS_SAMPLING_UP_SMOTE: 0/1 if minority class of dataset has been upsampled using SMOTE. 1 means 100% minority samples, 0.5 means half as many minority samples as majority samples
# - PROCESS_SAMPLING_UP_ADASYN: 0/1 if minority class of dataset has been upsampled using ADASYN
# - PROCESS_SAMPLING_DOWN_OSS: 0/1 if borderline samples have been dropped using one-sided selection
# - PROCESS_SAMPLING_DOWN_CNN: 0/1 if CNN has been used to downsample
# - PROCESS_SAMPLING_DOWN_TOMEK: 0/1 if samples part of tomek links have been dropped
# - PROCESS_WEIGHT: 0/1 if trainingsamples are weighted inversly the number of instances
# - PROCESS_SCALE_MINORITY: 0/1 if minority class samples have been scaled
# - PROCESS_SCALE_MODE: "fixed" (normalization constant explicitly set), "auto" (scaled until no tomek links)
# - PROCESS_SCALE_C: The normalization constant, in case SCALE_MODE is set to "fixed"
