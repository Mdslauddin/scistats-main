import math

def gamma_distribution(x, alpha, beta):
    """
    Calculates the probability density function (PDF) of the gamma distribution
    at a given value x with shape parameter alpha and scale parameter beta.
    """
    # Check for valid input values
    if x <= 0 or alpha <= 0 or beta <= 0:
        return 0

    # Compute the PDF
    pdf = (beta**alpha / math.gamma(alpha)) * x**(alpha-1) * math.exp(-beta*x)
    return pdf
