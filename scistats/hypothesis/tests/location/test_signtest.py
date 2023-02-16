import numpy as np

def sign_test(x, y, alpha=0.05):
    """
    Performs the Sign test on two related samples.

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

    # Calculate the number of positive and negative differences
    n_pos = np.sum(d > 0)
    n_neg = np.sum(d < 0)

    # Calculate the number of ties
    n_ties = len(d) - n_pos - n_neg

    # Calculate the test statistic
    n = n_pos + n_neg
    k = np.min([n_pos, n_neg])
    p = 0.5 ** n_ties
    T = binom(n, p).cdf(k - 1) + binom(n, p).sf(k)

    # Calculate the critical value
    z_alpha = norm.ppf(1 - alpha / 2)
    z_crit = np.abs(z_alpha)

    # Calculate the p-value
    p = 2 * norm.cdf(-np.abs(T))

    # Determine whether to reject the null hypothesis
    reject_null = np.abs(T) > z_crit

    # Return the results
    result = {
        "test_statistic": T,
        "p_value": p,
        "reject_null": reject_null,
    }

    return result
