import math
from scipy.special import erf

def noncentral_t_distribution(x, df, lambda_):
    """
    Calculates the probability density function (PDF) of the Non-Central T distribution
    at a given value x with degrees of freedom df and non-centrality parameter lambda.
    """
    # Check for valid input values
    if x < 0 or df <= 0 or lambda_ < 0:
        return 0

    # Compute the PDF using the Non-Central T formula
    pdf = ((math.exp(-lambda_/2) * (1 + (x**2 / df))) / (math.sqrt(df) * (1 + erf(x / math.sqrt(2))))) / ((math.pi * df)**0.5)
    return pdf
