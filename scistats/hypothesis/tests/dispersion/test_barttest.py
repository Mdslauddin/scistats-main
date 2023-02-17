import numpy as np
from scipy.stats import chi2

def bartlett_test(*samples):
    """
    Conducts Bartlett's test to compare the variances of two or more samples.
    
    Parameters:
    *samples (array-like): Two or more samples to compare.
    
    Returns:
    test_statistic (float): The Bartlett's test statistic.
    p_value (float): The p-value of the test.
    """
    # Get the number of samples and the total number of observations
    k = len(samples)
    n = sum(len(sample) for sample in samples)
    
    # Calculate the mean and variance of each sample
    means = [np.mean(sample) for sample in samples]
    variances = [np.var(sample, ddof=1) for sample in samples]
    
    # Calculate the pooled variance and degrees of freedom
    pooled_variance = np.sum((len(sample) - 1) * variances) / (n - k)
    degrees_of_freedom = k - 1
    
    # Calculate the test statistic
    numerator = (n - k) * np.log(pooled_variance) - np.sum((len(sample) - 1) * np.log(variance) for sample, variance in zip(samples, variances))
    denominator = 1 + (1 / (3 * (k - 1))) * np.sum((len(sample) - 1) / np.log(variance / pooled_variance) for sample, variance in zip(samples, variances))
    test_statistic = numerator / denominator
    
    # Calculate the p-value using the chi-squared distribution
    p_value = chi2.sf(test_statistic, degrees_of_freedom)
    
    return test_statistic, p_value
