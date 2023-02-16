import numpy as np
from scipy.stats import rankdata

def wilcoxon_signed_rank_test(x, y, alpha=0.05):
    """
    Performs the Wilcoxon signed-rank test on two related samples.

    Parameters
    ----------
    x : array-like
        The first sample.
    y : array-like
        The second sample.
    alpha : float, optional
        The significance level (default is 0.05).

    Returns
    -------
    result : dict
        A dictionary containing the test statistic, p-value, and whether the null hypothesis is rejected.
    """

    # Check that the samples have the same length
    if len(x) != len(y):
        raise ValueError("The two samples must have the same length.")

    # Calculate the differences
    d = np.array(x) - np.array(y)

    # Calculate the absolute values of the differences and the ranks
    abs_d = np.abs(d)
    ranks = rankdata(abs_d)

    # Calculate the signed ranks
    signed_ranks = np.sign(d) * ranks

    # Calculate the sum of the positive and negative ranks
    pos_sum = np.sum(signed_ranks[signed_ranks > 0])
    neg_sum = np.sum(signed_ranks[signed_ranks < 0])

    # Calculate the test statistic
    T = np.min([pos_sum, neg_sum])

    # Calculate the critical value
    n = len(x)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_crit = z_alpha * np.sqrt(n * (n + 1) / 6)

    # Calculate the p-value
    p = 2 * norm.cdf(-np.abs(T) / np.sqrt(n * (n + 1) * (2*n + 1) / 6))

    # Determine whether to reject the null hypothesis
    reject_null = np.abs(T) > z_crit

    # Return the results
    result = {
        "test_statistic": T,
        "p_value": p,
        "reject_null": reject_null,
    }

    return result
