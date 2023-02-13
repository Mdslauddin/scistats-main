import math

def normal_distribution(x, mean, stddev):
    """
    Calculates the probability density function (PDF) of the Normal (Gaussian) distribution
    at a given value x with mean and standard deviation.
    """
    # Check for valid input values
    if stddev <= 0:
        return 0

    # Compute the PDF using the Normal (Gaussian) formula
    pdf = (1 / (math.sqrt(2 * math.pi) * stddev)) * math.exp(-0.5 * ((x - mean) / stddev)**2)
    return pdf
