#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:31:32 2025

@author: maximilianschulten
"""
import numpy as np

import matplotlib.pyplot as plt
from pprint import pprint

# Set a seed for random state
np.random.seed(0)
# Store # of points within unit ball
l1_results = []

for d in np.arange(start = 2, stop = 11, step = 1 ):
    # Generate 1000 x d matrix 
    data = np.random.uniform(-1, 1, (1000, d))
    # Find norm of each row of the amtrix
    distances = np.linalg.norm(data, axis=1, ord=1)
    # Count number of elements with norm <= 1
    n_in_unit_ball = sum(distances <= 1)
    l1_results.append([d, n_in_unit_ball])
    
l2_results = []
for d in np.arange(start = 2, stop = 11, step = 1 ):
    # Generate 1000 x d matrix 
    data = np.random.uniform(-1, 1, (1000, d))
    # Find norm of each row of the amtrix
    distances = np.linalg.norm(data, axis=1, ord=2)
    # Count number of elements with norm <= 1
    n_in_unit_ball = sum(distances <= 1)
    l2_results.append([d, n_in_unit_ball])
    
    
l3_results = []
for d in np.arange(start = 2, stop = 11, step = 1 ):
    # Generate 1000 x d matrix 
    data = np.random.uniform(-1, 1, (1000, d))
    # Find norm of each row of the amtrix
    distances = np.linalg.norm(data, axis=1, ord=3)
    # Count number of elements with norm <= 1
    n_in_unit_ball = sum(distances <= 1)
    l3_results.append([d, n_in_unit_ball])

l1_results = np.array(l1_results)
l2_results = np.array(l2_results)
l3_results = np.array(l3_results)
plt.figure(figsize=(6,4))
plt.plot(l1_results[:,0], l1_results[:,1], marker='o')
plt.plot(l2_results[:,0], l2_results[:,1], marker='x')
plt.plot(l3_results[:,0], l3_results[:,1], marker='+')
plt.title('Number of Points <= 1 From the Origin')
plt.xlabel('Dimension (d)')
plt.ylabel('Points in the Unit Ball')
plt.grid(True)
plt.show()