__all__ = ['gmean','hmean','pmean','kurtosis','moment','skew','kstat', 'kstatvar', 'tmean', 'tvar']

import numpy as np

#define custom function
def gmean(x):
    a = np.log(x)
    return np.exp(a.mean())


def hmean(x):
    """
    Calculate the harmonic mean 
    """
    sums = 0
    for els in x:
        sums += 1 / els 
        res = len(x)/sums 
    return res 