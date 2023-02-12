def geometric_probability(p, k):
    """
    Calculates the probability of observing k failures before the first success in a sequence of Bernoulli trials with success probability p.
    
    Arguments:
        p -- the success probability of each Bernoulli trial, a float in the range (0, 1).
        k -- the number of failures, a positive integer.
    
    Returns:
        The probability of observing k failures before the first success in a sequence of Bernoulli trials with success probability p.
    """
    return (1 - p)**(k - 1) * p
