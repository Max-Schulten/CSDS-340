#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 14:49:34 2025

@author: maximilianschulten
"""
# Utils
import pandas as pd
from itertools import product
import numpy as np
from math import inf
# Imputation
from sklearn.impute import SimpleImputer
# Splitting
from sklearn.model_selection import train_test_split
# MCAR
from mcar import generate_mcar_data
# Tree
from sklearn.tree import DecisionTreeClassifier
# Accuracy
from sklearn.metrics import accuracy_score


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
# Split into feature matrix and class vector
X, y = df.iloc[:,:-1].to_numpy(dtype=float), df.iloc[:,-1].to_numpy(dtype=float)
# Generate randomly missing
X_mcar = generate_mcar_data(X, 0.2, random_state=1)
# Split
X_train, X_test, y_train, y_test = train_test_split(X_mcar, y, random_state=1, test_size=0.5, stratify=y)

# Set of different imputation strategies
impute_strategies = ['median', 'mean', 'most_frequent']
# Set different hyperparams
impurity_metrics = ['gini', 'entropy', 'log_loss']
max_depths = list(range(2,13)) + [None]
hyperparams = list(product(impute_strategies, impurity_metrics, max_depths))

# Store best
best_result = {"accuracy": -inf}

for params in hyperparams:
    # Retrieve strategy & Impurity metric
    impute_strategy = params[0]
    impurity_metric = params[1]
    max_depth = params[2]
    # Prepare & Fit imputer to training and test data
    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    # Fit Decision tree 
    dtree = DecisionTreeClassifier(criterion=impurity_metric, max_depth=max_depth, random_state=1).fit(X_train_imputed, y_train)
    # Predict and compute accuracy
    preds = dtree.predict(X_test_imputed)
    acc = accuracy_score(y_test, preds)
    # Compare and store conditionally
    if acc > best_result['accuracy']:
        best_result = {
            'accuracy': acc,
            'impute_strategy': impute_strategy,
            'impurity_metric': impurity_metric,
            'max_depth': max_depth
            }
    
# Print results
print(
    "-"*80+"\n"+
    "BEST ACCURACY:\n" +
    f"Impute Strategy={best_result['impute_strategy']},\n" +
    f"Impurity Metric={best_result['impurity_metric']},\n" +
    f"Max Depth={best_result['max_depth']},\n" +
    f"Accuracy={best_result['accuracy']}\n"+
    "-"*80
)
