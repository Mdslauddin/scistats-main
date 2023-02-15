from scipy.stats import lilliefors

def lilliefors_test(data):
    """
    Performs the Lilliefors test on a given dataset.

    Parameters:
        data (ndarray): Array of data.

    Returns:
        tuple: A tuple containing the test statistic and the p-value for the test.
    """
    test_stat, p_value = lilliefors(data, method='asymptotic')
    return test_stat, p_value
