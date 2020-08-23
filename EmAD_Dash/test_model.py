from joblib import load
import numpy as np
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score, confusion_matrix
from time import time

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
