"""

Hypergeometric distribution

https://en.wikipedia.org/wiki/Hypergeometric_distribution

"""

import math

def hypergeometric_probability(N, K, n, k):
    """
    Calculates the probability of observing k successes in n trials from a population of N items with K successes.
    
    Arguments:
        N -- the size of the population, a positive integer.
        K -- the number of successful items in the population, a positive integer.
        n -- the number of trials, a positive integer.
        k -- the number of successful trials, a positive integer.
    
    Returns:
        The probability of observing k successes in n trials from a population of N items with K successes.
    """
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)
