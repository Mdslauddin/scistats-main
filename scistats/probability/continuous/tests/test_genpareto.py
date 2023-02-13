import math

def gp_distribution(x, mu, sigma, kappa):
    """
    Calculates the probability density function (PDF) of the Generalized Pareto (GP) distribution
    at a given value x with location parameter mu, scale parameter sigma, and shape parameter kappa.
    """
    # Check for valid input values
    if x < mu or sigma <= 0 or kappa >= 0:
        return 0

    # Compute the PDF using the GP formula
    pdf = kappa / sigma * ((x - mu) / sigma)**(-kappa-1)
    return pdf
