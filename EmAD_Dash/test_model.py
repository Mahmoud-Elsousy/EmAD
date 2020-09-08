from joblib import load
import numpy as np
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score, confusion_matrix
from time import time

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

def test_model(mod):

    xte = load('xte.joblib')
    yte = load('yte.joblib')
    t=time()
    scores = mod.clf.decision_function(xte) 
    inference_time = ((time() - t)*1000)/np.shape(xte)[0]
    auc = roc_auc_score(yte, scores)
    pan = precision_n_scores(yte, scores)
    y_pre = mod.clf.predict(xte)
    tn, fp, fn, tp = confusion_matrix(yte,y_pre).ravel()
    print('Inference time (stored:measured):  (%.3f:%.3f)'%(mod.inference_time,inference_time))
    print('AUC Score(stored:measured):  (%.3f:%.3f)'%(mod.auc,auc))
    print('P@n Score(stored:measured):  (%.3f:%.3f)'%(mod.pan,pan))
    print('TN (stored:measured):  (%d:%d)'%(mod.tn,tn))
    print('FP (stored:measured):  (%d:%d)'%(mod.fp,fp))
    print('FN (stored:measured):  (%d:%d)'%(mod.fn,fn))
    print('TP (stored:measured):  (%d:%d)'%(mod.tp,tp))

model = load('model.joblib')

test_model(model)
