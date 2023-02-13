import numpy as np 

def beta_distribution(alpha, beta, size=10000):
    """
    # Define the parameters for the beta distribution
    alpha = 2
    beta = 5

    # Generate random samples from the beta distribution
    samples = beta_distribution(alpha, beta)

    # Plot the histogram of the samples
    plt.hist(samples, bins=50, density=True)
    plt.show()
    """
    return np.random.beta(alpha, beta, size)
   