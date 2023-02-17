import numpy as np
from scipy.stats import norm, rankdata

def ansari_bradley(x, y):
    """
    Conducts the Ansari-Bradley test to compare the variances of two samples.
    
    Parameters:
    x (array-like): First sample.
    y (array-like): Second sample.
    
    Returns:
    test_statistic (float): The Ansari-Bradley test statistic.
    p_value (float): The p-value of the test.
    """
    n = len(x)
    m = len(y)
    
    # Calculate the ranks of the combined data
    ranks = rankdata(np.concatenate((x, y)))
    ranks_x = ranks[:n]
    ranks_y = ranks[n:]
    
    # Calculate the test statistic
    ab = np.sum(np.outer(ranks_x, ranks_y), axis=0)
    test_statistic = ab / (n * m * (n + m + 1) / 12)
    
    # Calculate the p-value using a normal approximation
    var_ab = np.sum(np.outer(ranks_x, ranks_y)**2, axis=0) - (n * m * (n + m + 1)**2 / 4)
    z_score = test_statistic / np.sqrt(var_ab)
    p_value = 2 * norm.sf(abs(z_score))
    
    return test_statistic, p_value
