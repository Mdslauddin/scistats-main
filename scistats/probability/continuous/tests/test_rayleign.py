import math

def rayleigh_distribution(x, scale):
    """
    Calculates the probability density function (PDF) of the Rayleigh distribution
    at a given value x with scale parameter.
    """
    # Check for valid input values
    if x < 0 or scale <= 0:
        return 0

    # Compute the PDF using the Rayleigh formula
    pdf = (x / (scale**2)) * math.exp(-x**2 / (2 * (scale**2)))
    return pdf
