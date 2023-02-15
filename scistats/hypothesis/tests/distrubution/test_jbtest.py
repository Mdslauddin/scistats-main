from scipy.stats import jarque_bera

def jarque_bera_test(data):
    """
    Performs the Jarque-Bera test on a given dataset.

    Parameters:
        data (ndarray): Array of data.

    Returns:
        tuple: A tuple containing the test statistic and the p-value for the test.
    """
    test_stat, p_value = jarque_bera(data)
    return test_stat, p_value
