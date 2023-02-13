import numpy as np

def stable_distribution(alpha, beta, gamma, delta, x):
    """
    Calculates the probability density function (PDF) of the Stable distribution
    at a given value x with parameters alpha, beta, gamma, and delta.
    """
    # Check for valid input values
    if alpha <= 0 or alpha > 2 or abs(beta) > 1:
        return 0

    # Compute the PDF using the Stable formula
    pdf = np.exp(-np.power(np.abs((x - delta) / gamma), alpha)) / (np.power(np.abs((x - delta) / gamma), alpha) * gamma * np.power(np.pi, 0.5) * np.power(2 - alpha, 0.5))
    pdf *= np.power(np.cos((np.pi * alpha * beta / 2) + (beta * np.arctan(np.tan(np.pi * alpha / 2))) / alpha), -1 / alpha)

    return pdf
