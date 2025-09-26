#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 10:51:11 2025

@author: maximilianschulten
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from pprint import pprint

# Set a seed for random state
np.random.seed(0)
# Store # of points within unit ball
unit_ball_results = []
# Store nearest neighbor results
nn_results = []
for d in np.arange(start = 2, stop = 11, step = 1 ):
    # Generate 1000 x d matrix 
    data = np.random.uniform(-1, 1, (1000, d))
    # Find norm of each row of the amtrix
    distances = np.linalg.norm(data, axis=1)
    # Count number of elements with norm <= 1
    n_in_unit_ball = sum(distances <= 1)
    # Append results
    unit_ball_results.append([d, n_in_unit_ball])
    # Find mean pairwise distance
    mu_dist = np.mean(pdist(data))
    # Find nearest neighbor
    # Need to ask for 2 neighbors since first neighbor is point itself
    nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(data)
    # Find euclidean distances of points to nearest neighbor
    distances, indices = nbrs.kneighbors(data)
    # Find mean 1 nearest neighbor distance
    mu_nn_dist = np.mean(distances[:,1])
    # Store results
    nn_results.append([d, mu_nn_dist/mu_dist])
    # print(f"Dimension: {d}, # of Pts. in Unit Ball {n_in_unit_ball}, Mean Pairwise Distance: {mu_dist}, Mean 1 NN Distance: {mu_nn_dist}")
    
    

unit_ball_results = np.array(unit_ball_results)
nn_results = np.array(nn_results)
pprint(unit_ball_results)
pprint(nn_results)
plt.figure(figsize=(6,4))
plt.plot(unit_ball_results[:,0], unit_ball_results[:,1], marker='o')
plt.title('Number of Points <= 1 From the Origin')
plt.xlabel('Dimension (d)')
plt.ylabel('Points in the Unit Ball')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(nn_results[:,0], nn_results[:,1], marker='o')
plt.title('Mean Distance of 1st NN (Scaled by Average Pairwise Distance)\nvs. Dimension of Data')
plt.xlabel('Dimension (d)')
plt.ylabel('Scaled 1 NN Euclidean Distance')
plt.grid(True)
plt.show()