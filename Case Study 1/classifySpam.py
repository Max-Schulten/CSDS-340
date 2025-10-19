#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:36:53 2025

@author: maximilianschulten
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.model_selection import train_test_split

# Define a model
model = make_pipeline(
        SimpleImputer(missing_values=-1, strategy='median'),
        StandardScaler(),
        
        LogisticRegression(max_iter=999,
                           C=0.5, 
                           random_state=1, 
                           penalty='elasticnet',
                           solver='saga',
                           l1_ratio=0.5)
    )

# Cross validation strategy
cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=1)


# Preprocessing strategy
pipeline = Pipeline([
    ("imputer", SimpleImputer(missing_values=-1, strategy='median')),
    ("standardize", StandardScaler()),
    ("classify", GaussianNB())
    ])


def aucCV(features,labels):

    scores = cross_val_score(model,features,labels,cv=10,scoring='roc_auc')
    
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    
    model = make_pipeline(
        SimpleImputer(missing_values=-1, strategy="median"),
        StandardScaler(),
        BaggingClassifier(
            estimator=KNeighborsClassifier(
                n_neighbors=9,
                metric="minkowski",
                p=1,
                weights="distance"
            ),
            n_estimators=623,
            n_jobs=-1,
            max_features=0.17802172282792486,
            max_samples=0.6771376951247073,
            bootstrap=True,
            bootstrap_features=False,
            random_state=420,
        )
    )
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
    