import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def exponential_distribution(rate, size=10000):
    return np.random.exponential(1/rate, size)

# Define the parameters for the exponential distribution
rate = 2

# Generate random samples from the exponential distribution
samples = exponential_distribution(rate)

# Plot the histogram of the samples
plt.hist(samples, bins=50, density=True)

# Plot the theoretical PDF of the exponential distribution
x = np.linspace(0, 5, 1000)
pdf = rate * np.exp(-rate * x)
plt.plot(x, pdf, 'r', linewidth=2)

plt.show()
