import math

def weibull_distribution(x, alpha, beta):
    """
    Calculates the probability density function (PDF) of the Weibull distribution
    at a given value x with shape parameter alpha and scale parameter beta.
    """
    # Check for valid input values
    if alpha <= 0 or beta <= 0:
        return 0
    
    # Compute the PDF using the Weibull distribution formula
    if x >= 0:
        pdf = (alpha / beta) * ((x / beta) ** (alpha - 1)) * (math.exp(-((x / beta) ** alpha)))
    else:
        pdf = 0
    return pdf
