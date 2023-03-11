"""

Bernoulli distribution
https://en.wikipedia.org/wiki/Bernoulli_distribution


"""
#https://en.wikipedia.org/wiki/Bernoulli_distribution
# https://towardsdatascience.com/bernoulli-distribution-with-python-from-scratch-89fda3c822b

import numpy as np
#created a bernoulli class
 
class bernoulli():
    def pmf(x,p):
        """
        probability mass function        
        """
        f = p**x*(1-p)**(1-x)
        return f
    
    def mean(p):
        """
        expected value of bernoulli random variable
        """
        return p
    
    def var(p):
        """
        variance of bernoulli random variable
        """
        return p*(1-p)
    
    def std(p):
        """
        standart deviation of bernoulli random variable
        """
        return bernoulli.var(p)**(1/2)
    
    def rvs(p,size=1):
        """
        random variates
        """
        rvs = np.array([])
        for i in range(0,size):
            if np.random.rand() <= p:
                a=1
                rvs = np.append(rvs,a)
            else:
                a=0
                rvs = np.append(rvs,a)
        return rvs