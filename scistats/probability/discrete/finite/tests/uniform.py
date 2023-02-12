
"""
Discrete uniform distribution
https://en.wikipedia.org/wiki/Discrete_uniform_distribution

"""
import random

def uniform_distribution(a, b, n):
    """
    Generates n random numbers with a uniform distribution between a and b.
    """
    result = []
    for i in range(n):
        result.append(random.uniform(a, b))
    return result

print(uniform_distribution(1, 10, 5))


def discrete_uniform_cdf(x, a, b):
    """
    Calculates the cumulative distribution function (CDF) for a discrete uniform distribution
    with minimum value 'a' and maximum value 'b' at point 'x'.
    """
    if x < a:
        return 0
    elif x >= a and x <= b:
        return (x - a + 1) / (b - a + 1)
    else:
        return 1

    
def discrete_uniform_pdf(x, a, b):
    """
    Calculates the probability density function (PDF) for a discrete uniform distribution
    with minimum value 'a' and maximum value 'b' at point 'x'.
    """
    if x >= a and x <= b:
        return 1.0 / (b - a + 1)
    else:
        return 0

    
def discrete_uniform_inv_cdf(p, a, b):
    """
    Calculates the inverse cumulative distribution function (CDF) for a discrete uniform distribution.
    
    Arguments:
        p -- the probability, a float in the range [0, 1].
        a -- the lower bound of the distribution, an integer.
        b -- the upper bound of the distribution, an integer.
    
    Returns:
        The value x such that the CDF of the discrete uniform distribution at x is equal to p.
    """
    return a + int((b - a + 1) * p)

def discrete_uniform_mean(a, b):
    """
    Calculates the mean of a discrete uniform distribution.
    
    Arguments:
        a -- the lower bound of the distribution, an integer.
        b -- the upper bound of the distribution, an integer.
    
    Returns:
        The mean of the discrete uniform distribution.
    """
    return (a + b) / 2

def discrete_uniform_variance(a, b):
    """
    Calculates the variance of a discrete uniform distribution.
    
    Arguments:
        a -- the lower bound of the distribution, an integer.
        b -- the upper bound of the distribution, an integer.
    
    Returns:
        The variance of the discrete uniform distribution.
    """
    return (b - a + 1)**2 / 12


import random

def random_discrete_uniform(a, b):
    """
    Generates a random number from a discrete uniform distribution.
    
    Arguments:
        a -- the lower bound of the distribution, an integer.
        b -- the upper bound of the distribution, an integer.
    
    Returns:
        A random number from the discrete uniform distribution.
    """
    return random.randint(a, b)

