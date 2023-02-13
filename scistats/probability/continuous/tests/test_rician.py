import math

def rician_distribution(x, mean, stddev):
    """
    Calculates the probability density function (PDF) of the Rician distribution
    at a given value x with mean and standard deviation.
    """
    # Check for valid input values
    if stddev <= 0:
        return 0

    # Compute the PDF using the Rician formula
    pdf = (x / (stddev**2)) * math.exp(-(x**2 + mean**2) / (2 * (stddev**2))) * math.i0(x * mean / (stddev**2))
    return pdf
