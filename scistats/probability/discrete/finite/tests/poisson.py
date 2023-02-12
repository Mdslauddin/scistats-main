import math

def poisson_probability(lam, k):
    """
    Calculates the probability of observing k events in a Poisson process with rate lam.
    
    Arguments:
        lam -- the rate of the Poisson process, a positive float.
        k -- the number of events, a non-negative integer.
    
    Returns:
        The probability of observing k events in a Poisson process with rate lam.
    """
    return math.exp(-lam) * (lam**k) / math.factorial(k)
