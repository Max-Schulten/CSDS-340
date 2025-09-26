#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:25:36 2025

@author: maximilianschulten
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

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

# Extract features and labels
X, y = df.iloc[:,:-1].to_numpy(dtype=float), df.iloc[:,-1].to_numpy(dtype=float)

# Split 50/50 with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.5, stratify=y)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


"""
Logistic Regression with an L2 penalty on the raw features.
"""
LR_raw = LogisticRegression(penalty='l2', random_state=1, max_iter=500)
LR_raw.fit(X_train, y_train)
preds_LR_raw = LR_raw.predict(X_test)
LR_acc = accuracy_score(y_test, preds_LR_raw)
print(f"Raw Logistic Regression Score: {LR_acc}")

"""
LDA
"""
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
preds_lda = lda.predict(X_test)
LDA_acc = accuracy_score(y_test, preds_lda)

print(f"LDA Score: {LDA_acc}")

"""
PCA + Logistic Regression
"""
best_acc, best_k = -1, None
for k in range(1, X_train.shape[1]+1):
    pca = PCA(n_components=k, random_state=1)
    Xtr = pca.fit_transform(X_train)
    Xte = pca.transform(X_test)
    clf = LogisticRegression(penalty="l2", random_state=1, max_iter=1000)
    clf.fit(Xtr, y_train)
    acc = accuracy_score(y_test, clf.predict(Xte))
    if acc > best_acc:
        best_acc, best_k = acc, k
print(f"PCA + LR best: acc={best_acc}, k={best_k}")


