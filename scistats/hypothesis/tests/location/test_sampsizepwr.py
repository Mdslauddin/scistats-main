import numpy as np
from scipy.stats import norm

def calculate_sample_size(effect_size, alpha, power, ratio=1):
    """
    Calculates the required sample size for a two-sample t-test.

    Parameters
    ----------
    effect_size : float
        The standardized effect size (difference between the means divided by the standard deviation).
    alpha : float
        The significance level (Type I error rate).
    power : float
        The desired power of the test (1 - Type II error rate).
    ratio : float, optional
        The ratio of sample sizes between the two groups (default is 1, indicating equal sample sizes).

    Returns
    -------
    n : float
        The required sample size per group.
    """

    # Calculate the noncentrality parameter
    ncp = effect_size * np.sqrt(ratio) / np.sqrt(1 + ratio)

    # Calculate the critical value
    z_alpha = norm.ppf(1 - alpha / 2)

    # Calculate the critical value under the alternative hypothesis
    z_beta = norm.ppf(power)

    # Calculate the required sample size per group
    n = ((z_alpha + z_beta)**2 * (1 + ratio) / effect_size**2) / (1 + 1/ratio)

    return n


def calculate_power(n1, n2, effect_size, alpha, ratio=1):
    """
    Calculates the power of a two-sample t-test.

    Parameters
    ----------
    n1 : int
        The sample size of the first group.
    n2 : int
        The sample size of the second group.
    effect_size : float
        The standardized effect size (difference between the means divided by the standard deviation).
    alpha : float
        The significance level (Type I error rate).
    ratio : float, optional
        The ratio of sample sizes between the two groups (default is 1, indicating equal sample sizes).

    Returns
    -------
    power : float
        The power of the test (1 - Type II error rate).
    """

    # Calculate the noncentrality parameter
    ncp = effect_size * np.sqrt(ratio) / np.sqrt(1 + ratio)

    # Calculate the critical value under the null hypothesis
    t_alpha = norm.ppf(1 - alpha / 2)

    # Calculate the critical value under the alternative hypothesis
    t_beta = ncp / np.sqrt(1/n1 + 1/n2)

    # Calculate the power of the test
    power = norm.cdf(t_alpha - t_beta)

    return power
