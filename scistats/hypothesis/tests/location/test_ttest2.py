import numpy as np
from scipy.stats import t

def two_sample_ttest(x, y, alpha=0.05, equal_var=True):
    """
    Performs a two-sample t-test on two independent samples.

    Parameters
    ----------
    x : array-like
        The first sample.
    y : array-like
        The second sample.
    alpha : float, optional
        The significance level (default is 0.05).
    equal_var : bool, optional
        Whether to assume equal variances for the two populations (default is True).

    Returns
    -------
    result : dict
        A dictionary containing the test statistic, p-value, and whether the null hypothesis is rejected.
    """

    # Calculate the sample sizes, means, and standard deviations
    n1 = len(x)
    n2 = len(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)

    # Calculate the pooled standard deviation if the variances are assumed to be equal
    if equal_var:
        s_pooled = np.sqrt(((n1-1)*x_std**2 + (n2-1)*y_std**2) / (n1+n2-2))

        # Calculate the t-value
        t_val = (x_mean - y_mean) / (s_pooled * np.sqrt(1/n1 + 1/n2))

        # Calculate the degrees of freedom
        df = n1 + n2 - 2

    # Calculate the separate standard deviations if the variances are not assumed to be equal
    else:
        t_val, p_val = ttest_ind(x, y, equal_var=False)

        # Calculate the degrees of freedom
        df = np.floor((x_std**2/n1 + y_std**2/n2)**2 / 
                      ((x_std**2/n1)**2 / (n1-1) + (y_std**2/n2)**2 / (n2-1)))

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
