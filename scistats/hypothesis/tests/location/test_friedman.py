import numpy as np
from scipy.stats import rankdata

def friedman_test(*args):
    """
    Performs the Friedman's test on a set of data.

    Args:
        *args: Arrays or lists of the same length containing the data.

    Returns:
        A tuple containing the test statistic and the corresponding p-value.
    """
    k = len(args)  # number of groups
    n = len(args[0])  # number of samples per group
    
    # Calculate ranks
    data = np.concatenate(args)
    ranks = rankdata(data).reshape((k, n))

    # Calculate average ranks
    avg_ranks = np.mean(ranks, axis=0)

    # Calculate the Friedman's test statistic
    ss_total = np.sum(np.square(np.sum(ranks, axis=0)))
    ss_groups = n * np.sum(np.square(avg_ranks - np.mean(avg_ranks)))
    chi2 = (ss_total - ss_groups) / (k * (k - 1))

    # Calculate the p-value
    df = k - 1
    p_value = 1 - stats.chi2.cdf(chi2, df)

    return chi2, p_value
