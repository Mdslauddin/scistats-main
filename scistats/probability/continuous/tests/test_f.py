import math
import scipy.special

def f_distribution(x, dfn, dfd):
    """
    Calculates the cumulative density function (CDF) of the F-distribution
    at a given value x with degrees of freedom parameters dfn and dfd.
    """
    # Compute the CDF using the regularized beta function
    numer = scipy.special.betainc(dfn/2, dfd/2, dfn*x/(dfn*x + dfd))
    denom = scipy.special.beta(dfn/2, dfd/2)
    return numer/denom
