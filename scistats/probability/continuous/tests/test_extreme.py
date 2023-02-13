import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def extreme_value_distribution(loc, scale, size=10000):
    return np.random.standard_extreme_value(loc=loc, scale=scale, size=size)

# Define the parameters for the extreme value distribution
loc = 0
scale = 1

# Generate random samples from the extreme value distribution
samples = extreme_value_distribution(loc, scale)

# Plot the histogram of the samples
plt.hist(samples, bins=50, density=True)

# Plot the theoretical PDF of the extreme value distribution
x = np.linspace(-5, 5, 1000)
pdf = stats.genextreme.pdf(x, loc=loc, scale=scale)
plt.plot(x, pdf, 'r', linewidth=2)

plt.show()
