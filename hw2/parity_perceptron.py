#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 15:07:55 2025

@author: maximilianschulten
"""

import numpy as np
from perceptron_logic import Perceptron
import matplotlib.pyplot as plt
from pprint import pprint

X = np.array(
    [
     [0, 0, 0],
     [0, 0, 1],
     [0, 1, 0],
     [0, 1, 1],
     [1, 0, 0],
     [1, 0, 1],
     [1, 1, 0],
     [1, 1, 1]
     ]
    )

y = np.array(
    [
     1,
     0,
     0,
     1,
     0,
     1,
     1,
     0
     ]
    )


# Plot the training space.
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap="bwr", s=100)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")

plt.show()

results = []
best_perceptron = None
# Fit perceptron for different learning rates to try and find best fit
for eta in np.logspace(0, -9, 10):
    perceptron = Perceptron(eta=eta, random_state=1, n_iter=50)
    fitted_perceptron = perceptron.fit(X, y)
    results.append((eta, fitted_perceptron.errors_[-1], fitted_perceptron.w_, fitted_perceptron.b_))
    
    if not best_perceptron:
        best_perceptron = perceptron
    elif best_perceptron.errors_[-1] > fitted_perceptron.errors_[-1]:
        best_perceptron = fitted_perceptron

results.sort(key=lambda x: x[1])

print(f"BEST ACCURACY ACHIEVE FOR ETA={float(results[0][0])}: {1-results[0][1]/X.shape[0]}")
print("Weights:")
pprint(results[0][2])
print("Bias:")
pprint(results[0][3])
