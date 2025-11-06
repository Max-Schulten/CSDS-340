#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 15:43:22 2025

@author: maximilianschulten
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# %% Data
train = np.loadtxt("trainPoly.csv", delimiter=",")
X_tr, y_tr = train[:, 0].reshape(-1,1), train[:, 1].reshape(-1,1)
test = np.loadtxt("testPoly.csv", delimiter=",")
X_te, y_te = test[:, 0].reshape(-1,1), test[:, 1].reshape(-1,1)

# %% Fit and Evaluate
train_mses = []
test_mses = []
degs = np.arange(1,10)
norm_weights = []

for deg in degs:
    model = make_pipeline(
        PolynomialFeatures(degree = deg),
        #LinearRegression()
        Ridge(alpha=1e-6,random_state=1)
        )
    model.fit(X=X_tr, y=y_tr)
    
    test_preds = model.predict(X_te)
    mse_test = mean_squared_error(y_te, test_preds)
    test_mses.append(mse_test)
    
    train_preds = model.predict(X_tr)
    mse_train = mean_squared_error(y_tr, train_preds)
    train_mses.append(mse_train)
    
    w = model[1].coef_
    norm_w = np.linalg.norm(w)
    norm_weights.append(
        np.power(norm_w, 2)/deg
        )
    
# %% Visualize MSE
plt.figure(figsize=(7, 4))
plt.plot(degs, train_mses, marker='x', label="Training Set")
plt.plot(degs, test_mses, marker="o", label="Test Set")
plt.xlabel("Polynomial Regression Degree")
plt.ylabel("Mean Squared Error on Test Set")
plt.legend()
plt.show()
plt.savefig("mse_deg_ridge.png", dpi=800.)

# %% Visualize Normalized Squared Magnitude of Weight Vector
plt.figure(figsize=(7, 4))
plt.yscale('log')
plt.plot(degs, norm_weights, marker='x')
plt.xlabel("Polynomial Regression Degree")
plt.ylabel("Normalized Squared Weight Vector Magnitude")
plt.show()
plt.savefig("w_mag_ridge.png", dpi=800.)