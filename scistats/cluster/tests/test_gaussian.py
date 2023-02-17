# Gaussian mixture model 

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis

def gaussian_mixture_cdf(x, weights, means, variances):
    cdf = np.zeros_like(x)
    for w, mu, var in zip(weights, means, variances):
        cdf += w * norm.cdf(x, mu, np.sqrt(var))
    return cdf




def cluster_gaussian_mixture(data, n_clusters):
    """
    import numpy as np

    # Generate random data from two Gaussians
    np.random.seed(42)
    n_samples = 100
    X = np.concatenate([np.random.normal(0, 1, size=(n_samples, 2)), np.random.normal(3, 1, size=(n_samples, 2))], axis=0)

    # Cluster data using Gaussian mixture model
    clusters = cluster_gaussian_mixture(X, n_clusters=2)

    # Print cluster assignments
    print(clusters)

    """
    # Fit Gaussian mixture model
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(data)

    # Predict cluster assignments
    clusters = gmm.predict(data)
    return clusters


def mahalanobis_to_gmm(point, component, covariance):
    # Compute Mahalanobis distance
    inv_covariance = np.linalg.inv(covariance)
    distance = mahalanobis(point, component, inv_covariance)

    return distance


import numpy as np

def gaussian_mixture_pdf(X, means, covariances, weights):
    n_components = len(weights)
    n_samples, n_features = X.shape

    pdf = np.zeros(n_samples)
    for i in range(n_components):
        mean = means[i]
        covariance = covariances[i]
        weight = weights[i]
        pdf += weight * multivariate_normal.pdf(X, mean=mean, cov=covariance)

    return pdf

