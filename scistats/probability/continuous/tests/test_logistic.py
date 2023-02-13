import math

def logistic_distribution(x, mu, s):
    """
    Calculates the probability density function (PDF) of the Logistic distribution
    at a given value x with location parameter mu and scale parameter s.
    """
    # Check for valid input values
    if s <= 0:
        return 0

    # Compute the PDF using the Logistic formula
    pdf = math.exp(-(x - mu) / s) / (s * (1 + math.exp(-(x - mu) / s))**2)
    return pdf
