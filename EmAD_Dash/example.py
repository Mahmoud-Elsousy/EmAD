from joblib import load
import numpy as np

# Define emadModel // Temp. When emad is packaged in pip this would not be needed.
class emadModel:
    def __init__(self, name, clf):
        self.name = name
        self.clf = clf
        self.isTrained = 'No'
        self.n_features = 0
        self.size=0
        self.training_time = 0
        self.inference_time = 0
        self.auc = 0
        self.pan = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.b1 = 0
        self.b10 = 0
        self.b100 = 0
        self.b1000 = 0

# Load the model
model = load('model.joblib')

# Provide data as numpy array(n_samples, n_features)
x_test = np.random.rand(5, model.n_features)#.reshape(-1, 1) #If data has a single feature

# Get results: 1 if the sample is an Anomaly, 0 if normal. 
y_pre = model.clf.predict(x_test)

for i in range(len(y_pre)):
    print(x_test[i],'\t Anomaly? ', y_pre[i])


