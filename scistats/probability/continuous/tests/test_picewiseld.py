def piecewise_linear_distribution(x, intervals, probabilities):
    """
    Calculates the probability density function (PDF) of a Piecewise Linear distribution
    at a given value x with intervals and corresponding probabilities.
    """
    # Check for valid input values
    if len(intervals) + 1 != len(probabilities):
        return 0

    # Initialize the PDF with 0
    pdf = 0

    # Compute the PDF using the Piecewise Linear formula
    for i in range(len(intervals)):
        if intervals[i][0] <= x < intervals[i][1]:
            pdf = probabilities[i] / (intervals[i][1] - intervals[i][0])
            break
        elif x == intervals[i][1]:
            pdf = probabilities[i+1]
            break

    return pdf
