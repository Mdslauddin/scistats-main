import math

def loguniform_distribution(x, a, b):
    """
    Calculates the probability density function (PDF) of the Log-Uniform distribution
    at a given value x with lower bound a and upper bound b.
    """
    # Check for valid input values
    if x <= 0 or a <= 0 or b <= 0 or x < a or x > b:
        return 0

    # Compute the PDF using the Log-Uniform formula
    pdf = 1 / (x * math.log(b / a))
    return pdf
