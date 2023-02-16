import numpy as np
from scipy.stats import rankdata

def kruskal_wallis(*args):
    """
    # Generate some example data
    a = np.random.normal(0, 1, 50)
    b = np.random.normal(1, 1, 50)
    c = np.random.normal(2, 1, 50)

    # Call the Kruskal-Wallis test
    H, p = kruskal_wallis(a, b, c)

    # Print the results
    print("Kruskal-Wallis test:")
    print("H = {:.4f}, p = {:.4f}".format(H, p))
    """
    # Combine all input arrays into a single 1D array
    combined = np.concatenate(args)
    
    # Compute the ranks of the combined data
    ranks = rankdata(combined, method='average')
    
    # Split the ranks back into separate arrays
    rank_arrays = np.split(ranks, np.cumsum([a.size for a in args])[:-1])
    
    # Compute the group sums of ranks
    group_sums = np.array([np.sum(r) for r in rank_arrays])
    
    # Compute the sample sizes
    n = np.array([a.size for a in args])
    
    # Compute the overall sample size
    N = np.sum(n)
    
    # Compute the mean rank of each group
    group_means = group_sums / n
    
    # Compute the total sum of squares
    T = np.sum((ranks - np.mean(ranks))**2)
    
    # Compute the between-group sum of squares
    B = np.sum((group_means - np.mean(ranks))**2 * n)
    
    # Compute the within-group sum of squares
    W = T - B
    
    # Compute the degrees of freedom
    df_b = len(args) - 1
    df_w = N - len(args)
    df_t = N - 1
    
    # Compute the test statistic
    H = (B / df_b) / (W / df_w)
    
    # Compute the p-value
    p = 1 - scipy.stats.chi2.cdf(H, df_b)
    
    return H, p
