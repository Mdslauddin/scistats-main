import math
from scipy.special import iv, gammainc

def noncentral_chi_square_distribution(x, k, lambda_):
    """
    Calculates the probability density function (PDF) of the Non-Central Chi-Squared distribution
    at a given value x with degrees of freedom k and non-centrality parameter lambda.
    """
    # Check for valid input values
    if x < 0 or k <= 0 or lambda_ < 0:
        return 0

    # Compute the PDF using the Non-Central Chi-Squared formula
    pdf = iv(k/2-1, math.sqrt(lambda_)*math.sqrt(x/2)) * (math.exp(-lambda_/2) / 2**(k/2)) * (x/2)**(k/2-1) / iv(k/2, math.sqrt(lambda_))
    return pdf
