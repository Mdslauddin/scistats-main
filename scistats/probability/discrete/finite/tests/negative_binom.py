import math

def negative_binomial_probability(p, r, k):
    """
    Calculates the probability of observing k failures before the r-th success in a sequence of Bernoulli trials with success probability p.
    
    Arguments:
        p -- the success probability of each Bernoulli trial, a float in the range (0, 1).
        r -- the number of successes, a positive integer.
        k -- the number of failures, a non-negative integer.
    
    Returns:
        The probability of observing k failures before the r-th success in a sequence of Bernoulli trials with success probability p.
    """
    return math.comb(k + r - 1, r - 1) * (1 - p)**k * p**r
