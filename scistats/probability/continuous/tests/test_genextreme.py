import math

def gev_distribution(x, mu, sigma, kappa):
    """
    Calculates the probability density function (PDF) of the Generalized Extreme Value (GEV) distribution
    at a given value x with location parameter mu, scale parameter sigma, and shape parameter kappa.
    """
    # Check for valid input values
    if sigma <= 0:
        return 0

    # Compute the standardized value z
    z = (x - mu) / sigma

    # Compute the PDF using the GEV formula
    if kappa == 0:
        pdf = math.exp(-math.exp(-z)) / sigma
    elif kappa < 0:
        pdf = (math.exp(-((1 + kappa*z)**(-1/kappa)))/sigma) / ((-kappa)*(1 + kappa*z)**((1/kappa) + 1))
    else:
        pdf = (math.exp(-(1 + kappa*z)**(-1/kappa))) / (sigma * (1 + kappa*z)**((1/kappa)))

    return pdf
