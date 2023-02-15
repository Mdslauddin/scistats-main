from scipy.stats import kstest

def kolmogorov_smirnov_test(data, cdf):
    """
    Performs the one-sample Kolmogorov-Smirnov test on a given dataset.

    Parameters:
        data (ndarray): Array of data.
        cdf (callable): Cumulative distribution function to test against.

    Returns:
        tuple: A tuple containing the test statistic and the p-value for the test.
    """
    test_stat, p_value = kstest(data, cdf)
    return test_stat, p_value
