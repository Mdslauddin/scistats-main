import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def burr_type_xii_distribution(c, k, size=10000):
    return (1 + (c * np.random.pareto(k, size))**(-1/c))**(-1/k)

# Define the parameters for the Burr Type XII distribution
c = 1
k = 1

# Generate random samples from the Burr Type XII distribution
samples = burr_type_xii_distribution(c, k)

# Plot the histogram of the samples
plt.hist(samples, bins=50, density=True)

# Plot the theoretical PDF of the Burr Type XII distribution
x = np.linspace(0, 5, 1000)
pdf = c * k * (1 + x**c)**(-k-1) * x**(c-1)
plt.plot(x, pdf, 'r', linewidth=2)

plt.show()
