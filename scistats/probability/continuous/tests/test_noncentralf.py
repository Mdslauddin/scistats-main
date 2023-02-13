import math
from scipy.special import betainc, gammainc

def noncentral_f_distribution(x, dfn, dfd, lambda_):
    """
    Calculates the probability density function (PDF) of the Non-Central F distribution
    at a given value x with numerator degrees of freedom dfn and denominator degrees of freedom dfd, 
    and non-centrality parameter lambda.
    """
    # Check for valid input values
    if x < 0 or dfn <= 0 or dfd <= 0 or lambda_ < 0:
        return 0

    # Compute the PDF using the Non-Central F formula
    pdf = ((dfn * lambda_)**(dfn/2) * (dfd + dfn * x)**(-(dfn + dfd)/2) * math.sqrt(dfd / x)) / (x * betainc(dfn/2, dfd/2, (dfn * lambda_) / (dfd + dfn * x)))
    return pdf
