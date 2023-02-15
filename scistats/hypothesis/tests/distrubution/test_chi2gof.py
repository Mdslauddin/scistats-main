import numpy as np
from scipy.stats import chi2

def chi_square_test(observed, expected):
    """
    Performs the Chi-square goodness-of-fit test on a given set of observed and expected values.

    Parameters:
        observed (ndarray): Array of observed values.
        expected (ndarray): Array of expected values.

    Returns:
        tuple: A tuple containing the test statistic and the p-value for the test.
    """
    n = len(observed)
    dof = n - 1
    residuals = observed - expected
    chi_squared = np.sum(residuals**2 / expected)
    p_value = 1 - chi2.cdf(chi_squared, dof)
    return chi_squared, p_value
