import numpy as np

def kde(x, data, bandwidth=None):
    """
    Calculates the kernel density estimate (KDE) at a given value x using a given dataset and bandwidth.
    """
    # Number of samples in the dataset
    n = len(data)

    # Default bandwidth value
    if bandwidth is None:
        bandwidth = 1.06 * np.std(data) * n**(-0.2)

    # Compute the KDE estimate
    kde_estimate = 0
    for sample in data:
        kde_estimate += np.exp(-(x - sample)**2 / (2 * bandwidth**2))

    kde_estimate /= n * bandwidth * np.sqrt(2 * np.pi)
    return kde_estimate
