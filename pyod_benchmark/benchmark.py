# Edited from: https://github.com/yzhao062/pyod/blob/master/notebooks/benchmark.py

from __future__ import division
from __future__ import print_function

import os
import sys
from time import time

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.sod import SOD

from pyod.utils.utility import standardizer



# Define data file and read X and y
mat_file_list = ['arrhythmia.mat',
                 'cardio.mat',
                 'glass.mat',
                 'ionosphere.mat',
                 'letter.mat',
                 'lympho.mat',
                 'mnist.mat',
                 'musk.mat',
                 'optdigits.mat',
                 'pendigits.mat',
                 'pima.mat',
                 'satellite.mat',
                 'satimage-2.mat',
                 'shuttle.mat',
                 'vertebral.mat',
                 'vowels.mat',
                 'wbc.mat']

# define the number of iterations
n_ite = 2
n_classifiers = 10

df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc',
              'ABOD', 'CBLOF', 'FB', 'HBOS', 'IForest', 'KNN', 'LOF',
              'MCD', 'OCSVM', 'PCA']

# initialize the container for saving the results
time_df = pd.DataFrame(columns=df_columns)

start_time = time()
print('Start Time: ',start_time)
for j in range(len(mat_file_list)):

    mat_file = mat_file_list[j]
    mat = loadmat(os.path.join('data', mat_file))

    X = mat['X']
    y = mat['y'].ravel()
    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # construct containers for saving results
    time_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]

    time_mat = np.zeros([n_ite, n_classifiers])

    for i in range(n_ite):
        print("\n... Processing", mat_file, '...', 'Iteration', i + 1)
        random_state = np.random.RandomState(i)

        # 60% data for training and 40% for testing
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4, random_state=random_state)

        # standardizing data for processing
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        classifiers = {'Angle-based Outlier Detector (ABOD)': ABOD(
            contamination=outliers_fraction),
            'Cluster-based Local Outlier Factor': CBLOF(
                n_clusters=10,
                contamination=outliers_fraction,
                check_estimator=False,
                random_state=random_state),
            'Feature Bagging': FeatureBagging(contamination=outliers_fraction,
                                              random_state=random_state),
            'Histogram-base Outlier Detection (HBOS)': HBOS(
                contamination=outliers_fraction),
            'Isolation Forest': IForest(contamination=outliers_fraction,
                                        random_state=random_state),
            'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
            'Local Outlier Factor (LOF)': LOF(
                contamination=outliers_fraction),
            'Minimum Covariance Determinant (MCD)': MCD(
                contamination=outliers_fraction, random_state=random_state),
            'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
            'Principal Component Analysis (PCA)': PCA(
                contamination=outliers_fraction, random_state=random_state),
        }
        classifiers_indices = {
            'Angle-based Outlier Detector (ABOD)': 0,
            'Cluster-based Local Outlier Factor': 1,
            'Feature Bagging': 2,
            'Histogram-base Outlier Detection (HBOS)': 3,
            'Isolation Forest': 4,
            'K Nearest Neighbors (KNN)': 5,
            'Local Outlier Factor (LOF)': 6,
            'Minimum Covariance Determinant (MCD)': 7,
            'One-class SVM (OCSVM)': 8,
            'Principal Component Analysis (PCA)': 9,
        }

        for clf_name, clf in classifiers.items():
            t0 = time()
            clf.fit(X_train_norm)
            t1 = time()
            duration = round(t1 - t0, ndigits=4)

            time_mat[i, classifiers_indices[clf_name]] = duration

    time_list = time_list + np.mean(time_mat, axis=0).tolist()
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)


    # Save the results for each run
    time_df.to_csv('time.csv', index=False, float_format='%.3f')
end_time = time()
print('End Time: ',end_time)
print('Test Total Time: ', end_time - start_time)
