import math

def multinomial_probability(n, p, x):
    """
    Calculates the probability of observing x events in a multinomial experiment with n trials and probabilities p.
    
    Arguments:
        n -- the number of trials, a positive integer.
        p -- a list of probabilities, each a float in the range [0, 1]. The sum of p should be equal to 1.
        x -- a list of integers representing the number of occurrences of each event. The sum of x should be equal to n.
    
    Returns:
        The probability of observing x events in a multinomial experiment with n trials and probabilities p.
    """
    return math.factorial(n) / (math.factorial(x[0]) * math.factorial(x[1]) * ... * math.factorial(x[k])) * (p[0]**x[0]) * (p[1]**x[1]) * ... * (p[k]**x[k])
