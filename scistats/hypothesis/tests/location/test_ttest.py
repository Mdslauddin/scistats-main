import numpy as np
from scipy.stats import t

def one_sample_ttest(sample, popmean, alpha=0.05):
    """
    Performs a one-sample t-test on a sample.

    Parameters
    ----------
    sample : array-like
        The sample to be tested.
    popmean : float
        The population mean to be tested against.
    alpha : float, optional
        The significance level (default is 0.05).

    Returns
    -------
    result : dict
        A dictionary containing the test statistic, p-value, and whether the null hypothesis is rejected.
    """
    # Calculate sample size, mean and standard deviation
    n = len(sample)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)

    # Calculate the t-value
    t_val = (sample_mean - popmean) / (sample_std / np.sqrt(n))

    # Calculate the degrees of freedom
    df = n - 1

    # Calculate the critical value
    t_crit = t.ppf(1 - alpha / 2, df)

    # Calculate the p-value
    p_val = 2 * t.sf(np.abs(t_val), df)

    # Determine whether to reject the null hypothesis
    reject_null = np.abs(t_val) > t_crit

    # Return the results
    result = {
        "test_statistic": t_val,
        "p_value": p_val,
        "reject_null": reject_null,
    }

    return result


def paired_ttest(x, y, alpha=0.05):
    """
    Performs a paired-sample t-test on two related samples.

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

    # Calculate the sample mean and standard deviation of the differences
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)

    # Calculate the t-value
    t_val = d_mean / (d_std / np.sqrt(len(d)))

    # Calculate the degrees of freedom
    df = len(d) - 1

    # Calculate the critical value
    t_crit = t.ppf(1 - alpha / 2, df)

    # Calculate the p-value
    p_val = 2 * t.sf(np.abs(t_val), df)

    # Determine whether to reject the null hypothesis
    reject_null = np.abs(t_val) > t_crit

    # Return the results
    result = {
        "test_statistic": t_val,
        "p_value": p_val,
        "reject_null": reject_null,
    }

    return result
