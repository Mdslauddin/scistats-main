# Nearest Neighbors 

__all__ = ['knnsearch', 'rangesearch']
import numpy as np

def knnsearch(X, Q, k):
    """
    find_k_nearest_neighbors
    
    X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
    Q = np.array([[2, 2], [3, 3]])
    k = 2

    neighbors = find_k_nearest_neighbors(X, Q, k)

    print("Neighbors:", neighbors)

    """
    # Compute the distance between each query point and each input point
    dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Q**2, axis=1) - 2 * np.dot(X, Q.T)
    # Find the indices of the k nearest neighbors for each query point
    indices = np.argsort(dists, axis=0)[:k, :]
    # Extract the k nearest neighbors from the input data
    neighbors = np.take(X, indices, axis=0)
    return neighbors



def rangesearch(X, Q, r):
    """
    find_neighbors_within_distance
    X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
    Q = np.array([[2, 2], [3, 3]])
    r = 2

    neighbors = find_neighbors_within_distance(X, Q, r)

    print("Neighbors:", neighbors)

    """
    # Compute the distance between each query point and each input point
    dists = np.sqrt(np.sum((X - Q[:, np.newaxis])**2, axis=2))
    # Find the indices of the neighbors within the specified distance for each query point
    indices = np.argwhere(dists <= r)
    # Extract the neighbors from the input data
    neighbors = np.take(X, indices[:, 1], axis=0)
    return neighbors

