import numpy as np
from scipy.stats import f

def equal_var_test(*samples):
    """
    Conducts multiple-sample tests to determine if samples have equal variances.
    
    Parameters:
    *samples (arrays): The samples to test.
    
    Returns:
    test_statistic (float): The F test statistic.
    p_value (float): The p-value of the test.
    """
    # Calculate the sample variances and degrees of freedom
    variances = []
    dfs = []
    for sample in samples:
        n = len(sample)
        var = np.var(sample, ddof=1)
        variances.append(var)
        dfs.append(n - 1)
    
    # Calculate the F test statistic
    mean_var = np.mean(variances)
    ssw = np.sum([(n - 1) * var for (n, var) in zip(dfs, variances)])
    sse = np.sum([(n - 1) * (var - mean_var) for (n, var) in zip(dfs, variances)])
    test_statistic = (ssw / (len(samples) - 1)) / (sse / (np.sum(dfs) - len(samples)))
    
    # Calculate the p-value using the F distribution
    df1 = len(samples) - 1
    df2 = np.sum(dfs) - len(samples)
    p_value = f.sf(test_statistic, df1, df2) * 2
    
    return test_statistic, p_value
