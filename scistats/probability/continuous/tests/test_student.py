import math
import scipy.special as sps

def student_t_distribution(x, degrees_of_freedom):
    """
    Calculates the probability density function (PDF) of the Student's t-distribution
    at a given value x with degrees of freedom.
    """
    # Check for valid input values
    if degrees_of_freedom <= 0:
        return 0

    # Compute the PDF using the Student's t-distribution formula
    pdf = (math.gamma((degrees_of_freedom + 1) / 2) / (math.sqrt(degrees_of_freedom * math.pi) * math.gamma(degrees_of_freedom / 2))) * (1 + ((x**2) / degrees_of_freedom))**(-(degrees_of_freedom + 1) / 2)
    return pdf
