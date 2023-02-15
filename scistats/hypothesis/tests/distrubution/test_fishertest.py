from scipy.stats import fisher_exact

def fisher_exact_test(data):
    """
    Performs Fisher's exact test on a 2x2 contingency table.

    Parameters:
        data (ndarray): A 2x2 contingency table.

    Returns:
        tuple: A tuple containing the odds ratio and the p-value for the test.
    """
    odds_ratio, p_value = fisher_exact(data)
    return odds_ratio, p_value
