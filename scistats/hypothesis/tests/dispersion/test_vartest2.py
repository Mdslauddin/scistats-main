import numpy as np
from scipy.stats import f

def f_test(sample1, sample2):
    """
    Conducts the two-sample F-test to determine if two samples have equal variances.
    
    Parameters:
    sample1 (array-like): The first sample.
    sample2 (array-like): The second sample.
    
    Returns:
    test_statistic (float): The F test statistic.
    p_value (float): The p-value of the test.
    """
    # Calculate the sample variances and degrees of freedom
    n1 = len(sample1)
    n2 = len(sample2)
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    df1 = n1 - 1
    df2 = n2 - 1
    
    # Calculate the F test statistic
    test_statistic = var1 / var2 if var1 > var2 else var2 / var1
    
    # Calculate the p-value using the F distribution
    p_value = f.sf(test_statistic, df1, df2) * 2
    
    return test_statistic, p_value
