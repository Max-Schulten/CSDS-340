#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 12:20:55 2025

@author: maximilianschulten
"""

# Model
from sklearn.neighbors import KNeighborsClassifier
# Splitting
from sklearn.model_selection import train_test_split
# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Imputation
from sklearn.impute import SimpleImputer
# Feature selection
from sklearn.feature_selection import SequentialFeatureSelector
# Testing
from sklearn.metrics import accuracy_score
# Utilities
import pandas as pd
import numpy as np
from math import inf

class IdentityScaler:
    def transform(self, X):
        return X

cols = [
    "preg",
    "glu",
    "bp",
    "skin",
    "insu",
    "bmi",
    "dpf",
    "age",
    "class"
]

# Read in data
df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=cols)
# Prepare imputer
#imputer = SimpleImputer(missing_values=0, strategy='most_frequent')
# Impute columns with obviously missing data 
#cols_to_impute = ["glu", "bp", "skin", "insu", "bmi"]
#df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

# Extract features and labels
X, y = df.iloc[:,:-1].to_numpy(dtype=float), df.iloc[:,-1].to_numpy(dtype=float)

# Split 50/50 with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.5, stratify=y)


"""
FIND BEST COMBINATION OF K, FEATURES, SCALING
"""
standScaler = StandardScaler().fit(X_train)
minMaxScaler = MinMaxScaler().fit(X_train)
scalers = [IdentityScaler(), standScaler, minMaxScaler]
ks = np.arange(start=2, stop=20, step=1)

best_accuracy = {"accuracy": -inf}

for scaler in scalers:
    # Scale both training and testing data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for k in ks:
        nn_classifier = KNeighborsClassifier(n_neighbors=k)

        # evaluate ALL FEATURES
        all_idx = np.arange(X_train.shape[1])
        Xtr_all = X_train_scaled[:, all_idx]
        Xte_all = X_test_scaled[:, all_idx]
        nn_classifier.fit(Xtr_all, y_train)
        preds_all = nn_classifier.predict(Xte_all)
        acc_all = accuracy_score(y_test, preds_all)

        #  evaluate SFS 
        sfs = SequentialFeatureSelector(
            estimator=nn_classifier,
            direction='forward'
        )
        sfs.fit(X_train_scaled, y_train)
        sfs_idx = sfs.get_support(indices=True)
        Xtr_sfs = X_train_scaled[:, sfs_idx]
        Xte_sfs = X_test_scaled[:, sfs_idx]
        nn_classifier.fit(Xtr_sfs, y_train)
        preds_sfs = nn_classifier.predict(Xte_sfs)
        acc_sfs = accuracy_score(y_test, preds_sfs)

        # choose local winner between ("all", "sfs")
        if acc_sfs >= acc_all:
            local = {
                "k": k,
                "scaler": scaler.__class__.__name__,
                "feature_mode": "sfs",
                "selected_features": ', '.join([cols[i] for i in sfs_idx]),
                "accuracy": acc_sfs
            }
        else:
            local = {
                "k": k,
                "scaler": scaler.__class__.__name__,
                "feature_mode": "all",
                "selected_features": ', '.join([cols[i] for i in all_idx]),
                "accuracy": acc_all
            }

        # compare local winner to global best
        if local["accuracy"] > best_accuracy["accuracy"]:
            best_accuracy = local

# Print results
print(
    "-"*80+"\n"+
    f"BEST ACCURACY: K={best_accuracy['k']},\n" +
    f"Feature Mode={best_accuracy['feature_mode']},\n" +
    f"Selected Features={best_accuracy['selected_features']},\n" +
    f"Scaler={best_accuracy['scaler']},\n" +
    f"Accuracy={best_accuracy['accuracy']}\n"+
    "-"*80
)