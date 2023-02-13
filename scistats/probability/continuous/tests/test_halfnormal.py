import math

def half_normal_distribution(x, mu, sigma):
    """
    Calculates the probability density function (PDF) of the half-normal distribution
    at a given value x with location parameter mu and scale parameter sigma.
    """
    # Check for valid input values
    if x < 0 or sigma <= 0:
        return 0

    # Compute the standardized value z
    z = (x - mu) / sigma

    # Compute the PDF using the half-normal formula
    pdf = math.sqrt(2/math.pi) / sigma * math.exp(-z**2 / 2)
    return pdf
