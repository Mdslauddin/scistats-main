def uniform_distribution_continuous(x, lower, upper):
    """
    Calculates the probability density function (PDF) of the continuous uniform distribution
    at a given value x with lower and upper bounds.
    """
    # Check for valid input values
    if not (lower <= upper):
        return 0
    
    # Compute the PDF using the continuous uniform distribution formula
    if lower <= x <= upper:
        pdf = 1 / (upper - lower)
    else:
        pdf = 0
    return pdf
