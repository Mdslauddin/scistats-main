def triangular_distribution(x, left, mode, right):
    """
    Calculates the probability density function (PDF) of the triangular distribution
    at a given value x with left, mode and right parameters.
    """
    # Check for valid input values
    if not (left <= mode <= right):
        return 0
    
    # Compute the PDF using the triangular distribution formula
    if x < left:
        pdf = 0
    elif x < mode:
        pdf = (2 * (x - left)) / ((right - left) * (mode - left))
    elif x < right:
        pdf = (2 * (right - x)) / ((right - left) * (right - mode))
    else:
        pdf = 0
    return pdf
