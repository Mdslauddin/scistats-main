import math

def nakagami_distribution(x, mu, omega):
    """
    Calculates the probability density function (PDF) of the Nakagami distribution
    at a given value x with shape parameter mu and spread parameter omega.
    """
    # Check for valid input values
    if x < 0 or mu <= 0 or omega <= 0:
        return 0

    # Compute the PDF using the Nakagami formula
    pdf = (2 * mu**mu / math.gamma(mu)) * (x / omega)**(2 * mu - 1) * math.exp(-mu * (x / omega)**2)
    return pdf
