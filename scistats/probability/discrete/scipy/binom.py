# https://www.tertiaryinfotech.com/compute-binomial-distribution-using-python-from-scratch/
# https://towardsdatascience.com/binomial-distribution-and-binomial-test-in-python-statistics-pyshark-91aa6403d674
"""

Binomial distribution
https://en.wikipedia.org/wiki/Binomial_distribution

"""
# Negative Binomial as defined in
# https://mathworld.wolfram.com/NegativeBinomialDistribution.html
# https://stackoverflow.com/questions/63839778/binomial-distribution-simulation-python

from cmath import sqrt
class binom:
    
    def  pmf(k,n,p):
        
        return math.comb(n,k)*(p**k)*((1-p)**(n-k))

    def mean(n,p):
        return (n*p)

    def std(n,p):
        
        """
        Calculates the standard deviation
        :return std:
        """
        std = sqrt(n * p * (1-p))
        return std