

__all__ = ['describe', 'gmean', 'hmean', 'pmean', 'kurtosis', 'mode', 'moment', 'expectile', 'skew', 'kstat',
           'kstatvar', 'tmean', 'tvar', 'tmin', 'tmax', 'tstd', 'tsem', 'variation', 'find_repeats', 'trim_mean', 
           'gstd', 'iqr', 'sem', 'beyes_mvs', 'mvsdist', 'entropy', 'differential_entropy', 'median_abs_deviation', 'median_abs_deviation'] 



import numpy as np
# https://www.geeksforgeeks.org/scipy-stats-gmean-function-python/
# https://www.statology.org/geometric-mean-python/
#define custom function
def gmean(x):
    a = np.log(x)
    return np.exp(a.mean())

# https://en.wikipedia.org/wiki/Harmonic_mean#Weighted_harmonic_mean
def hmean(x):
    """
    Calculate the harmonic mean 
    """
    sums = 0
    for els in x:
        sums += 1 / els 
        res = len(x)/sums 
    return res


# https://eli.thegreenplace.net/2009/03/21/efficient-integer-exponentiation-algorithms

from scipy.stats import mode
import numpy as np
import statistics
def skews(x):
    x =statistics.mean(x) - statistics.mode(x)
    std = np.std(x)
    #x = std/x
    return std

# https://www.turing.com/kb/calculating-skewness-and-kurtosis-in-python
from scipy.stats import mode
import numpy as np
def skews(x):
    x =np.mean(x) - mode(x)
    print(x)
    std = np.std(x)
    print(std)
    x = std/x
    return x