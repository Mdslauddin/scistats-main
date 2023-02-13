import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def birnbaum_saunders_distribution(beta, sigma, size=10000):
    mu = beta / np.sqrt(1 - beta**2)
    return np.random.lognormal(mean=mu, sigma=sigma, size=size)

# Define the parameters for the Birnbaum-Saunders distribution
beta = 0.5
sigma = 0.5

# Generate random samples from the Birnbaum-Saunders distribution
samples = birnbaum_saunders_distribution(beta, sigma)

# Plot the histogram of the samples
plt.hist(samples, bins=50, density=True)

# Plot the theoretical PDF of the Birnbaum-Saunders distribution
x = np.linspace(0, 10, 1000)
pdf = stats.lognorm.pdf(x, sigma, scale=np.exp(beta / np.sqrt(1 - beta**2)))
plt.plot(x, pdf, 'r', linewidth=2)

plt.show()
