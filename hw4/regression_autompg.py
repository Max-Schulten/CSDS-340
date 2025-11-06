#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 19:32:29 2025

@author: maximilianschulten
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold
from pprint import pprint

# %% Config
RS = 1
CV = KFold(n_splits=5, random_state=RS, shuffle=True) 
MAX_ITER = 999999999

# %% Data Loading
data = np.loadtxt('auto-mpg-missing-data-removed.txt', comments='"')
X = data[:,1:]
y = data[:,0]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=RS)

# %% Models
elastic = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=0.65, l1_ratio=0.96, random_state=RS, max_iter=MAX_ITER)
    )

lin_svr = make_pipeline(
    StandardScaler(),
    LinearSVR(random_state=1, max_iter=MAX_ITER)
    )

ker_svr = make_pipeline(
    StandardScaler(),
    SVR(max_iter=MAX_ITER)
    )

rf = make_pipeline(
    RandomForestRegressor(random_state=RS)
    )

knn = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor()
    )

models = {
    "elastic": elastic,
    "linear svr": lin_svr,
    "kernel svr": ker_svr,
    "random forest": rf,
    "knn": knn
    }

# %% Model Params
elastic_params = {
    "elasticnet__alpha": np.round(np.linspace(0.1, 1.5, 14), 2),
    "elasticnet__l1_ratio": np.round(np.linspace(0.1, 1, 19), 2)
    }

lin_svr_params = {
    "linearsvr__C": [0.1, 1, 10, 100, 1000]
    }

ker_svr_params = {
    "svr__kernel": ['poly', 'rbf'],
    "svr__degree": [2, 3, 4],
    "svr__C": [0.1, 1, 10, 100, 1000],
    "svr__gamma": [1, 0.1, 0.001, 0.0001]
    }

rf_params = {
    "randomforestregressor__n_estimators": [100, 200, 300 ,400],
    "randomforestregressor__max_depth": [2,4,6,8, None],
    "randomforestregressor__max_features": ["sqrt", "log2", None],
    "randomforestregressor__max_samples": [None, 0.2, 0.4, 0.6, 0.8]
    }

knn_params = {
    "kneighborsregressor__n_neighbors": [3, 5, 7, 9, 11, 13]
    }


params = {
    "elastic": elastic_params,
    "linear svr": lin_svr_params,
    "kernel svr": ker_svr_params,
    "random forest": rf_params,
    "knn": knn_params
    }

# %% Evaluate

for model_ in models.items():
    model_name = model_[0]
    model = model_[1]
    model_params = params[model_name]
    
    search = GridSearchCV(model, model_params, scoring="r2", cv=CV)
    
    search.fit(X_tr, y_tr)
    r2 = search.score(X_te, y_te)
    
    print(f"Model: {model_name}\tScore: {r2}")
    pprint(search.best_params_)
    print("-"*20)
    
    