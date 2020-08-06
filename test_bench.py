import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyod.utils.data import generate_data_clusters
from pyod.utils.data import generate_data

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

from joblib import dump, load

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

random_state = np.random.RandomState(3)
outliers_fraction = 0.1

# Generate Data
xtr, xte, ytr, yte = generate_data_clusters(n_train=1000, n_test=500, n_clusters=5,
                       n_features=3, contamination=0.1, size='same',
                       density='same', dist=0.25, random_state=random_state,
                       return_in_clusters=False)

print('Xtr Shape: ' + str(xtr.shape))
print('Ytr Shape: ' + str(ytr.shape))


# Define outlier detection methods to be compared
classifiers = {
    'Isolation Forest': IForest(contamination=outliers_fraction, random_state=random_state),
   
    'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
    
    'Average KNN': KNN(method='mean',contamination=outliers_fraction),
    
    'Local Outlier Factor (LOF)': LOF(n_neighbors=35, contamination=outliers_fraction),
}

# Fit the model
for i, (clf_name, clf) in enumerate(classifiers.items()):
    print('\n \n',i + 1, 'fitting', clf_name)
    # fit the data and tag outliers
    clf.fit(xtr)
#     scores_pred = clf.decision_function(xtr) * -1
    y_pred = clf.predict(xtr)
#     threshold = np.percentile(scores_pred, 100 * outliers_fraction)
#     print('Threshold: ', threshold)
    n_errors = (y_pred != ytr).sum()
    print('Erros Training Data: ', n_errors)
    
    yte_pred = clf.predict(xte)
    nte_errors = (yte_pred != yte).sum()
    print('Erros Testing Data: ', nte_errors)
    print('First 5 samples',yte_pred[:5] )
    
    # Serielize model
    dump(clf, str(i) + '.joblib')