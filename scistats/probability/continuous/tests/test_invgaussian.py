import math

def ig_distribution(x, mu, lambda_):
    """
    Calculates the probability density function (PDF) of the Inverse Gaussian (IG) distribution
    at a given value x with location parameter mu and shape parameter lambda.
    """
    # Check for valid input values
    if x <= 0 or mu <= 0 or lambda_ <= 0:
        return 0

    # Compute the PDF using the IG formula
    pdf = math.sqrt(lambda_ / (2 * math.pi * x**3)) * math.exp(-lambda_ * (x - mu)**2 / (2 * mu**2 * x))
    return pdf
