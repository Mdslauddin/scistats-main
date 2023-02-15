import numpy as np

def durbin_watson(residuals):
    """
    Performs the Durbin-Watson test on a given set of residuals.

    Parameters:
        residuals (ndarray): Array of residuals.

    Returns:
        float: The Durbin-Watson test statistic.
    """
    diff = np.diff(residuals)
    num = np.sum(diff**2)
    denom = np.sum(residuals**2)
    durbin_watson = num / denom
    return durbin_watson
