#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 19:12:45 2025

@author: maximilianschulten
"""
import numpy as np

def random_assign(n_examples, n_clusters):
    # List to store assignments
    assignments = np.zeros(n_examples, dtype=np.int64)
    
    # Map of used indices
    indices = np.arange(n_examples)
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Cluster Values, n_clusters integers
    clusters = np.arange(n_clusters, dtype=np.int64)
    
    # Assignment clusters in a round-robin fashion to a random unused index
    for i, idx in enumerate(indices):
        cluster = clusters[i % n_clusters] # Retrieve next assignment
        assignments[idx] = cluster

    return assignments, clusters

def distance_to_centroid(i, cluster, K, assignments):
    # Get examples in centroid
    idx = np.where(assignments == cluster)[0]
    j = idx.size
    
    term1 = K[i,i]
    term2 = (2.0/j)*K[i,idx].sum()
    term3 = (1.0/j**2)*np.sum(K[np.ix_(idx, idx)])
    
    return term1 - term2 + term3

def fit_kmeans_dot_products(K, n_clusters, max_iter=300):
    # Randomly assign examples to a cluster
    assignments, clusters = random_assign(K.shape[0], n_clusters)
    indices = np.arange(K.shape[0])
    
    # Main Loop
    for _ in range(max_iter):
        last_assignment = assignments.copy()
        new_assignments = np.empty_like(assignments)
        for i in indices:
            dists = [distance_to_centroid(i, cluster, K, assignments) for cluster in range(n_clusters)]
            new_assignments[i] = np.argmin(dists)
        if np.array_equal(new_assignments, last_assignment):
            return new_assignments
        
        assignments = new_assignments
    return assignments

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score
    import matplotlib.pyplot as plt
    
    X = []
    y = []
    
    centers = [[5,5], [0,0], [-5,5]]
    
    for idx, coords in enumerate(centers):
        cx = coords[0]
        cy = coords[1]
        y.extend([idx]*50)
        X.append(
            np.random.randn(50, 2) + np.array([cx, cy])
            )
    X = np.vstack(X)
        
    
    K = X @ X.T
    
    assignments = fit_kmeans_dot_products(K, n_clusters=3)
    ari = adjusted_rand_score(y, assignments)
    
    plt.scatter(X[:, 0], X[:, 1], c=assignments, s=40, alpha=0.7)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    
    print(f"ARI: {ari}")

    
    