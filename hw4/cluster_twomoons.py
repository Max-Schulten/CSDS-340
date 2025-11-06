#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 17:39:04 2025

@author: maximilianschulten
"""

# %% Setup
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# %% Config
MAX_ITER = 9999

# %% Data
data = np.loadtxt('twomoons.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

# %% K-Means
k_means = KMeans(n_clusters=2, random_state=1, max_iter=MAX_ITER)
k_means.fit(X)
ari = adjusted_rand_score(y, k_means.labels_)
print(f"K-Means ARI: {ari}")

#%% Spectral RBF
spectral = SpectralClustering(n_clusters=2, affinity='rbf', gamma=100, random_state=1)
spectral.fit(X)
ari = adjusted_rand_score(y, spectral.labels_)
print(f"Spectral RBF ARI: {ari}")

#%% Spectral NN
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=15, random_state=1)
spectral.fit(X)
ari = adjusted_rand_score(y, spectral.labels_)
print(f"Spectral NN ARI: {ari}")

# %% Hierarchical Sklearn
hier = AgglomerativeClustering(n_clusters=2, linkage='average')
hier.fit(X)
ari = adjusted_rand_score(y, hier.labels_)
print(f"Hierarchical (Sklearn) ARI: {ari}")

# %% Hierarchichal Sci-Py 
linked =  linkage(X, method='average')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.plot(y)
plt.show()
clusters = fcluster(linked, 1.5, criterion='distance')
ari = adjusted_rand_score(y, clusters)
print(f"Hierarchical (Sci-py) ARI: {ari}")