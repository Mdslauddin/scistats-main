def run_test(data):
    """
    Performs the Run test for randomness on a given dataset.

    Parameters:
        data (ndarray): Array of data.

    Returns:
        tuple: A tuple containing the test statistic and the p-value for the test.
    """
    n = len(data)
    runs = 1
    for i in range(1, n):
        if data[i] != data[i-1]:
            runs += 1
    exp_runs = (2 * n - 1) / 3
    var_runs = (16 * n - 29) / 90
    z = (runs - exp_runs) / (var_runs ** 0.5)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value
