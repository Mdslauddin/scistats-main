# Spectral Clustering 


import numpy as np
from sklearn.cluster import KMeans

def spectral_clustering(X, k, sigma, r):
    """
    
    X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
    k = 2
    sigma = 1
    r = 2

    labels = spectral_clustering(X, k, sigma, r)

    print("Labels:", labels)

    """
    # Compute the similarity matrix
    dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    S = np.exp(-dists / (2 * sigma**2))
    # Compute the Laplacian matrix
    D = np.diag(np.sum(S, axis=1))
    L = D - S
    # Compute the eigenvectors of the Laplacian matrix
    eigvals, eigvecs = np.linalg.eigh(L)
    # Take the r smallest eigenvectors
    V = eigvecs[:, :r]
    # Normalize the rows of V
    V = V / np.sqrt(np.sum(V**2, axis=1))[:, np.newaxis]
    # Cluster the rows of V using k-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(V)
    labels = kmeans.labels_
    return labels
