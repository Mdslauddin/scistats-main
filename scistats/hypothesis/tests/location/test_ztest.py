import math
from scipy.stats import norm

def z_score(sample_mean, pop_mean, sample_std, sample_size):
    """
    Calculates the z-score for a one-sample z-test.
    
    Args:
    sample_mean (float): the sample mean
    pop_mean (float): the population mean
    sample_std (float): the sample standard deviation
    sample_size (int): the sample size
    
    Returns:
    float: the z-score
    """
    return (sample_mean - pop_mean) / (sample_std / math.sqrt(sample_size))


def p_value(z_score, two_tailed=True):
    """
    Calculates the p-value for a one-sample z-test.
    
    Args:
    z_score (float): the z-score
    two_tailed (bool): whether to use a one-tailed or two-tailed test
    
    Returns:
    float: the p-value
    """
    if two_tailed:
        return 2 * (1 - norm.cdf(abs(z_score)))
    else:
        return 1 - norm.cdf(z_score)
