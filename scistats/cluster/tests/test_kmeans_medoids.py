# K-mean and K-medoids clustering 
import numpy as np

__all__ = ['kmeans','kmedoids','mahal']

def kmeans(data, k, max_iters=100):
    """
    data = np.array([[1, 2], [1, 4], [1, 0],
                 [4, 2], [4, 4], [4, 0]])
    k = 2
    labels, centroids = k_means(data, k)

    print("Labels:", labels)
    print("Centroids:", centroids)

    """
    # Initialize cluster centers randomly
    centroids = np.random.rand(k, data.shape[1])
    # Initialize the labels to all zeros
    labels = np.zeros(data.shape[0])
    # Loop for a maximum number of iterations
    for i in range(max_iters):
        # Assign each point to its nearest cluster center
        for j, point in enumerate(data):
            distances = np.linalg.norm(point - centroids, axis=1)
            labels[j] = np.argmin(distances)
        # Update the cluster centers to be the mean of the points in each cluster
        for c in range(k):
            points_in_cluster = data[labels == c]
            centroids[c] = np.mean(points_in_cluster, axis=0)
    return labels, centroids



def kmedoids(data, k, max_iters=100):
    """
    data = np.array([[1, 2], [1, 4], [1, 0],
                 [4, 2], [4, 4], [4, 0]])
    k = 2
    labels, medoids = k_medoids(data, k)

    print("Labels:", labels)
    print("Medoids:", medoids)

    """
    # Initialize medoids randomly
    medoids = np.random.choice(data.shape[0], size=k, replace=False)
    # Initialize the labels to all zeros
    labels = np.zeros(data.shape[0])
    # Loop for a maximum number of iterations
    for i in range(max_iters):
        # Assign each point to its nearest medoid
        for j, point in enumerate(data):
            distances = np.linalg.norm(data[medoids] - point, axis=1)
            labels[j] = np.argmin(distances)
        # Update the medoids to be the point that minimizes the average distance to all other points in the cluster
        for c in range(k):
            points_in_cluster = data[labels == c]
            if len(points_in_cluster) > 0:
                distances = np.linalg.norm(points_in_cluster - points_in_cluster[:, np.newaxis], axis=-1)
                costs = np.sum(distances, axis=1)
                medoids[c] = np.argmin(costs)
    return labels, medoids





def mahal(X, ref):
    """
    mahalanobis_distance
    
    X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
    ref = np.array([[1, 2], [1, 4], [1, 0]])

    dist = mahalanobis_distance(X, ref)

    print("Mahalanobis distances:", dist)

    """
    # Compute the inverse covariance matrix of the reference samples
    cov_inv = np.linalg.inv(np.cov(ref.T))
    # Compute the mean of the reference samples
    mean = np.mean(ref, axis=0)
    # Compute the Mahalanobis distance of each point in X to the reference samples
    dist = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        dist[i] = np.sqrt((x - mean).T @ cov_inv @ (x - mean))
    return dist
