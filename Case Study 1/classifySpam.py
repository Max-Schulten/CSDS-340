#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:36:53 2025

@author: maximilianschulten
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

model = make_pipeline(
    XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=2000,
        learning_rate=0.003,
        max_depth=5,
        min_child_weight=1,
        gamma=1,
        subsample=0.7,
        colsample_bytree=0.5333,
        reg_alpha=0.1841,
        reg_lambda=0.5957,
        scale_pos_weight=0.9,
        max_delta_step=2,
        n_jobs=-1,
        random_state=67,
        missing=-1
    )
)

def aucCV(features,labels):

    scores = cross_val_score(model,features,labels,cv=10,scoring='roc_auc')
    
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    
    
    model.fit(trainFeatures,trainLabels)
    
    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = model.predict_proba(testFeatures)[:,1]
    
    return testOutputs
    
# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data = np.loadtxt('spamTrain1.csv',delimiter=',')
    # Separate labels (last column)
    X = data[:,:-1]
    y = data[:,-1]
    
    """
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        stratify=y,
                                                        test_size=0.33,
                                                        random_state=1)
    """
    
    scores = aucCV(X, y)
    
    # Evaluating classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ",
          scores.mean())
    print("10-fold cross-validation AUC standard deviation:",
          scores.std())
    
    # Arbitrarily choose all odd samples as train set and all even as test set
    # then compute test set AUC for model trained only on fixed train set
    trainFeatures = X[0::2,:]
    trainLabels = y[0::2]
    testFeatures = X[1::2,:]
    testLabels = y[1::2]
    testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))
    
    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(2,1,1)
    plt.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.subplot(2,1,2)
    plt.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    plt.tight_layout()
    plt.show()
    