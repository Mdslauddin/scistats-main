import numpy as np
from scipy.stats import invwishart

def inverse_wishart_samples(df, scale_mat, num_samples):
    """
    Generates random samples from an Inverse Wishart distribution.

    Parameters:
        df (int): Degrees of freedom.
        scale_mat (ndarray): Scale matrix of the distribution.
        num_samples (int): Number of samples to generate.

    Returns:
        ndarray: Array of shape (num_samples, scale_mat.shape[0], scale_mat.shape[1])
            containing the generated samples.
    """
    inv_wishart = invwishart(df, scale=scale_mat)
    samples = inv_wishart.rvs(num_samples)
    return samples
