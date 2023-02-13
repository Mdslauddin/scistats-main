import math
import scipy.special as sps

def t_location_scale_distribution(x, location, scale, degrees_of_freedom):
    """
    Calculates the probability density function (PDF) of the t-location-scale distribution
    at a given value x with location, scale and degrees of freedom.
    """
    # Check for valid input values
    if degrees_of_freedom <= 0 or scale <= 0:
        return 0

    # Compute the PDF using the t-location-scale distribution formula
    pdf = (math.gamma((degrees_of_freedom + 1) / 2) / (math.sqrt(degrees_of_freedom * math.pi) * math.gamma(degrees_of_freedom / 2) * scale)) * (1 + (((x - location) / scale)**2) / degrees_of_freedom)**(-(degrees_of_freedom + 1) / 2)
    return pdf
