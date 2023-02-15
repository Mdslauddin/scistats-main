import numpy as np
from scipy.stats import norm

def anderson_darling_test(data, dist='norm'):
    """
    Performs the Anderson-Darling test for normality on a given dataset.

    Parameters:
        data (ndarray): Array of data to test.
        dist (str): Name of distribution to test against. Default is 'norm'.

    Returns:
        tuple: A tuple containing the test statistic and an array of critical values
            for a given significance level.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    mid_rank = np.arange(1, n+1) - 0.5
    ecdf = mid_rank / n
    cdf = getattr(norm, dist).cdf(sorted_data)
    ad_statistic = -n - np.sum((2 * np.arange(1, n+1) - 1) * (np.log(cdf) + np.log(1 - cdf[::-1])))
    ad_statistic *= (1 + 0.75/n + 2.25/n**2)
    critical_values = [0.576, 0.656, 0.787, 0.918, 1.092]
    if n <= 25:
        idx = n - 1
    else:
        idx = np.searchsorted(np.array([10, 25, 50, 100, 250, 500, 1000, 2500, 5000]), n)
    significance_level = [15, 10, 5, 2.5, 1][idx]
    return ad_statistic, critical_values[idx] / (1 + 4/n - 25/n**2), significance_level
