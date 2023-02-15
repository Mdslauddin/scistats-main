from scipy.stats import ks_2samp

def two_sample_kolmogorov_smirnov_test(data1, data2):
    """
    Performs the two-sample Kolmogorov-Smirnov test on two given datasets.

    Parameters:
        data1 (ndarray): Array of first data.
        data2 (ndarray): Array of second data.

    Returns:
        tuple: A tuple containing the test statistic and the p-value for the test.
    """
    test_stat, p_value = ks_2samp(data1, data2)
    return test_stat, p_value
