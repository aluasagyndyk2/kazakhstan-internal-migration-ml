"""Fixed model definitions and scoring helpers shared with Version 1.

Run run_analysis.py for the Version 2 protocol.
"""
from pathlib import Path
import json, platform, sys, datetime
import numpy as np
import pandas as pd
import sklearn, scipy, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,precision_score,
                            recall_score,roc_auc_score,average_precision_score,confusion_matrix)

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'; FIGURES=RESULTS/'figures'

def models():
    choices={
        'Majority dummy':DummyClassifier(strategy='most_frequent'),
        'Logistic Regression':LogisticRegression(C=1.0,max_iter=2000,random_state=42),
        'Decision Tree':DecisionTreeClassifier(max_depth=3,min_samples_leaf=5,random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=300,max_depth=4,min_samples_leaf=3,max_features='sqrt',random_state=42,n_jobs=1),
        'Gradient Boosting':GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,max_depth=2,min_samples_leaf=5,random_state=42)
    }
    result={}
    for name,model in choices.items():
        steps=[('impute',SimpleImputer(strategy='median'))]
        if name=='Logistic Regression': steps.append(('scale',StandardScaler()))
        result[name]=Pipeline(steps+[('model',model)])
    return result

def fit_model(model,name,X,y):
    # Same inverse-class-frequency sample weighting for all four learned models.
    args={} if name=='Majority dummy' else {'model__sample_weight':compute_sample_weight('balanced',y)}
    return model.fit(X,y,**args)

def metrics(y,pred,prob):
    cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    return dict(n=len(y),positive=int(np.sum(y)),accuracy=accuracy_score(y,pred),
                balanced_accuracy=balanced_accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro',zero_division=0),
                precision_positive=precision_score(y,pred,zero_division=0),recall_positive=recall_score(y,pred,zero_division=0),
                roc_auc=roc_auc_score(y,prob),average_precision=average_precision_score(y,prob),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
