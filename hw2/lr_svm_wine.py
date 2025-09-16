#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:09:09 2025

@author: maximilianschulten
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from pprint import pprint

# For readability
cols = [
    "class",
    "alcohol",
    "malic_acid",
    "ash",
    "ash_alc",      # alkalinity of ash
    "magnesium",
    "tot_phenols",
    "flav",
    "nonflav_phen",
    "proanth",
    "color_int",
    "hue",
    "od280_od315",
    "proline"
]

df = pd.read_csv("wine.data.csv", header= None, names=cols)

# Split into features (tabular), classes (vector)
X, y = df.iloc[:, 1:], df.iloc[:,0]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, 
                                                    random_state=1, stratify=y)
# Scale the data
X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)

LR_results = []
c_values = np.arange(0.1, 2, 0.1)

# Iterate over some C-values to find the best one
for c in c_values:
    # Fit and predict
    preds = LogisticRegression(random_state=1, C=c)\
        .fit(X_train_std, y_train)\
        .predict(X_test_std)
    
    # Retrieve accuracy
    acc = accuracy_score(y_test, preds)
    
    # Add to results
    LR_results.append({"C": c, "Accuracy": acc})

# Sort results by accuracy score
LR_results.sort(reverse=True, key=lambda x: x["Accuracy"])


# Print All results
pprint(LR_results)

print("-"*100)

# Print best C and Accuracy score
print(f"Logistic Regression (C={LR_results[0]['C']:.4f}): {LR_results[0]['Accuracy']:.4f}")

print("-"*100)

kernels = ['linear', 'poly', 'rbf', 'sigmoid'] # All kernel types
degrees = np.arange(1, 6, 1) # Degree of polynomial (Only applies to polynomial kernel)
gammas = np.arange(0, 1, 0.1)

SVC_results = []

for kernel in kernels:
    for c in c_values:
        for gamma in gammas:
            if kernel == "poly":
                for degree in degrees:
                    # Fit and predict
                    preds = SVC(random_state=1, C=c, kernel=kernel, degree=degree, gamma=gamma)\
                        .fit(X_train_std, y_train)\
                        .predict(X_test_std)# Note that degree will be ignored for non-polynomial kernel
                    
                    # Retrieve accuracy
                    acc = accuracy_score(y_test, preds)
                    
                    # Add to results
                    SVC_results.append({"C": c, "Kernel": kernel, "Degree": degree, "Gamma": gamma, "Accuracy": acc})
            else:
                # Fit and predict
                preds = SVC(random_state=1, C=c, kernel=kernel, gamma=gamma)\
                    .fit(X_train_std, y_train)\
                    .predict(X_test_std)# Note that degree will be ignored for non-polynomial kernel
                
                # Retrieve accuracy
                acc = accuracy_score(y_test, preds)
                
                # Add to results
                SVC_results.append({"C": c, "Kernel": kernel, "Degree": None, "Gamma": gamma, "Accuracy": acc})

# Sort results by accuracy score
SVC_results.sort(reverse=True, key=lambda x: x["Accuracy"])


# Print Top 5
pprint(SVC_results[0:4])

print("-"*100)

# Print best C, Kernel, Degree (if applicable) and Accuracy score
print(f"SVM Regression (Kernel={SVC_results[0]['Kernel']}, Degree={SVC_results[0]['Degree']}, C={SVC_results[0]['C']:.3f}, Gamma={SVC_results[0]['Gamma']:.43}): {SVC_results[0]['Accuracy']:.4f}")

print("-"*100)