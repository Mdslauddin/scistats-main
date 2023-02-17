import numpy as np
from scipy.stats import chi2

def chi2_var_test(sample, var):
    """
    Conducts the chi-square variance test to determine if a sample has a given variance.
    
    Parameters:
    sample (array-like): The sample to test.
    var (float): The hypothesized variance of the sample.
    
    Returns:
    test_statistic (float): The chi-square test statistic.
    p_value (float): The p-value of the test.
    """
    # Calculate the sample variance
    sample_var = np.var(sample, ddof=1)
    
    # Calculate the test statistic
    test_statistic = (len(sample) - 1) * sample_var / var
    
    # Calculate the p-value using the chi-squared distribution
    degrees_of_freedom = len(sample) - 1
    p_value = chi2.sf(test_statistic, degrees_of_freedom)
    
    return test_statistic, p_value
