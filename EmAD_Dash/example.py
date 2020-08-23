from joblib import load
import numpy as np

# Load the model
model = load('model.joblib')

# Provide data as numpy array(n_samples, n_features)
x_test = np.random.rand(5, model.n_features)#.reshape(-1, 1) #If data has a single feature

# Get results: 1 if the sample is an Anomaly, 0 if normal. 
y_pre = model.clf.predict(x_test)

for i in range(len(y_pre)):
    print(x_test[i],'\t Anomaly? ', y_pre[i])


