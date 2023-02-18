# Anomaly Detection 


__all__ = ['iforest','isanomaly','lof','isanomaly','ocsvm','isanomaly','robustcov', 'mahal', 'pdist2']


from sklearn.ensemble import IsolationForest



def iforest(X, n_estimators=100, contamination=0.1, random_state=None):
    """
    fit_isolation_forest
    import numpy as np

    # Generate some random data
    np.random.seed(0)
    X = np.random.randn(100, 2)

    # Fit the isolation forest model to the data
    model = fit_isolation_forest(X, n_estimators=100, contamination=0.1)

    # Predict the labels of the data points
    y_pred = model.predict(X)

    # Print the labels
    print("Labels:", y_pred)

    """
    # Fit the isolation forest model to the input data
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    model.fit(X)
    return model


def isanomaly(X, n_estimators=100, contamination=0.1, random_state=None):
    """
    find_anomalies_isolation_forest
    import numpy as np
    from sklearn.ensemble import IsolationForest

    # Generate some random data
    np.random.seed(0)
    X = np.random.randn(100, 2)

    # Find the anomalous data points using the isolation forest
    anomalies = find_anomalies_isolation_forest(X, n_estimators=100, contamination=0.1)

    # Print the anomalous data points
    print("Anomalies:", anomalies)

    """
    # Fit the isolation forest model to the input data
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    model.fit(X)

    # Predict the labels of the data points
    y_pred = model.predict(X)

    # Return the indices of the anomalous data points
    anomalies = np.argwhere(y_pred == -1).flatten()
    return anomalies


from sklearn.neighbors import LocalOutlierFactor

def lof(X, n_neighbors=20, contamination=0.1):
    """
    create_lof_model
    
    import numpy as np

    # Generate some random data
    np.random.seed(0)
    X = np.random.randn(100, 2)

    # Create the LOF model
    model, scores = create_lof_model(X, n_neighbors=20, contamination=0.1)

    # Print the scores
    print("LOF scores:", scores)

    """
    # Fit the LOF model to the input data
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = model.fit_predict(X)
    scores = model.negative_outlier_factor_

    # Return the fitted model and the scores
    return model, scores


import numpy as np
from sklearn.covariance import MinCovDet

def robustcov(X, alpha=0.5):
    """
    
    robust_cov_mean_estimate
    
    
    Calculate the robust multivariate covariance and mean estimate of a dataset X
    using the Minimum Covariance Determinant (MCD) algorithm.
    
    Args:
        X (numpy array): Set of n observations with m features.
        alpha (float, optional): Contamination parameter. Must be in the range (0, 0.5).
        
    Returns:
        cov (numpy array): m x m estimated covariance matrix.
        mean (numpy array): 1 x m estimated mean vector.
        
    # Generate a dataset with outliers
    X = np.random.rand(100, 5)
    X[:10] += 5

    # Calculate the robust multivariate covariance and mean estimate of X
    cov, mean = robust_cov_mean_estimate(X)

    print("Covariance matrix:")
    print(cov)
    print("Mean vector:")
    print(mean)

    """
    n, m = X.shape
    
    # Calculate the MCD estimate of the covariance matrix
    mcd = MinCovDet(assume_centered=False, support_fraction=1-alpha)
    mcd.fit(X)
    cov = mcd.covariance_
    
    # Calculate the MCD estimate of the mean vector
    mean = mcd.location_
    
    return cov, mean



import numpy as np
from scipy.spatial.distance import mahalanobis

def mahal(X, Y):
    """
    mahalanobis_distance
    
    Calculate Mahalanobis distance between a set of observations X and a
    reference set of observations Y.
    
    Args:
        X (numpy array): Set of n observations with m features.
        Y (numpy array): Reference set of n_ref observations with m features.
        
    Returns:
        dist (numpy array): n x n_ref matrix of Mahalanobis distances between X and Y.
        
        
    # Generate a set of observations
    X = np.random.rand(5, 3)

    # Generate a reference set of observations
    Y = np.random.rand(3, 3)

    # Calculate Mahalanobis distance between X and Y
    dist = mahalanobis_distance(X, Y)

    print(dist)

    """
    n, m = X.shape
    n_ref, m_ref = Y.shape
    
    if m != m_ref:
        raise ValueError("The number of features in X and Y must be the same.")
    
    # Calculate the inverse covariance matrix of Y
    cov = np.cov(Y.T)
    inv_cov = np.linalg.inv(cov)
    
    # Calculate Mahalanobis distance
    dist = np.zeros((n, n_ref))
    for i in range(n):
        for j in range(n_ref):
            dist[i, j] = mahalanobis(X[i], Y[j], inv_cov)
    
    return dist



def pdist2(X, Y):
    """
    pairwise_distance
    
    Calculate pairwise distance between two sets of observations X and Y
    using the Euclidean distance metric.
    
    Args:
        X (numpy array): Set of n1 observations with m features.
        Y (numpy array): Set of n2 observations with m features.
        
    Returns:
        dist (numpy array): n1 x n2 matrix of pairwise distances between X and Y.
        
    # Generate two sets of observations
    X = np.random.rand(5, 3)
    Y = np.random.rand(3, 3)

    # Calculate pairwise distance between X and Y
    dist = pairwise_distance(X, Y)

    print(dist)

    """
    n1, m = X.shape
    n2, m2 = Y.shape
    
    if m != m2:
        raise ValueError("The number of features in X and Y must be the same.")
    
    # Calculate pairwise distances
    dist = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            dist[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))
    
    return dist

