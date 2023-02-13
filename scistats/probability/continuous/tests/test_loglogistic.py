import math

def loglogistic_distribution(x, alpha, beta):
    """
    Calculates the probability density function (PDF) of the Log-Logistic distribution
    at a given value x with shape parameters alpha and beta.
    """
    # Check for valid input values
    if x <= 0 or alpha <= 0 or beta <= 0:
        return 0

    # Compute the PDF using the Log-Logistic formula
    pdf = (beta / alpha) * (x / alpha)**(beta - 1) / (1 + (x / alpha)**beta)**2
    return pdf
