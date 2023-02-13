import math

def lognormal_distribution(x, mu, sigma):
    """
    Calculates the probability density function (PDF) of the Log-Normal distribution
    at a given value x with location parameter mu and scale parameter sigma.
    """
    # Check for valid input values
    if x <= 0 or sigma <= 0:
        return 0

    # Compute the PDF using the Log-Normal formula
    pdf = (1 / (x * sigma * math.sqrt(2 * math.pi))) * math.exp(-((math.log(x) - mu)**2) / (2 * sigma**2))
    return pdf
