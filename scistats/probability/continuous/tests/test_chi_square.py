import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def chi_square_distribution(df, size=10000):
    return np.random.chisquare(df, size)

# Define the parameters for the chi-squared distribution
df = 5

# Generate random samples from the chi-squared distribution
samples = chi_square_distribution(df)

# Plot the histogram of the samples
plt.hist(samples, bins=50, density=True)

# Plot the theoretical PDF of the chi-squared distribution
x = np.linspace(0, 20, 1000)
pdf = stats.chi2.pdf(x, df)
plt.plot(x, pdf, 'r', linewidth=2)

plt.show()
