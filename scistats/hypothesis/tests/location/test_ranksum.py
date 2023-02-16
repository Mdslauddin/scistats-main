def wilcoxon_rank_sum_test(x, y):
    """
    Performs the Wilcoxon rank sum test on two samples x and y.
    """
    n1 = len(x)
    n2 = len(y)
    n = n1 + n2
    
    # Combine the samples and rank them
    ranks = rankdata(np.concatenate([x, y]))
    rank_x = ranks[:n1]
    rank_y = ranks[n1:]
    
    # Calculate the test statistic
    w = sum(rank_x) - n1 * (n1 + 1) / 2
    
    # Calculate the expected value and variance of the test statistic
    mu = n1 * n2 / 2
    var = n1 * n2 * (n1 + n2 + 1) / 12
    
    # Calculate the z-score
    z = (w - mu) / np.sqrt(var)
    
    # Calculate the p-value
    p = 2 * (1 - norm.cdf(abs(z)))
    
    return z, p
