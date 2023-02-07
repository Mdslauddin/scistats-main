""" 
Descriptive Statistics and Visualization 

Categories:
1.) Managing Data: Data import and export grouping variables
2.) Descriptive statistics : Numerical summarises and associated measures
3.) statistical visualizaiton : view data patterns and trens 

# Functions
## Central Tendency and Dispersion
- geomean 
- harmmean 
- trimean 
- kurtosis
- moment
- skewness

## Range, Deviation, and z-Score
- range	Range of values
- mad	Mean or median absolute deviation
- zscore  Standardized z-scores
"""
import math 
import numpy as np

__all__ = ['geomean','harmmean','trimean','kurtosis','moment','skewness']

def geomean(x):
    if type(x) == list:
        length = len(x)
        prod = math.prod(x)
        geom = pow(prod,(1/length))
    return geom 
    
    
def harmmean(numbers):
    sum_of_inverses = 0
    for num in numbers:
        sum_of_inverses += 1/num
    return len(numbers) / sum_of_inverses

# https://en.wikipedia.org/wiki/Trimean
def trimean(numbers):
    numbers.sort()
    n = len(numbers)
    q1 = numbers[n//4]
    q3 = numbers[3*n//4]
    median = numbers[n//2]
    return (q1 + 2 * median + q3) / 4




def kurtosis(numbers):
    """
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = kurtosis(numbers)
    print(result)

    """
    n = len(numbers)
    mean = np.mean(numbers)
    std = np.std(numbers)
    return sum(((x - mean) / std)**4 for x in numbers) / n - 3


def central_moment(numbers, k):
    """
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = central_moment(numbers, 3)
    print(result)
 
    
    """
    n = len(numbers)
    mean = np.mean(numbers)
    return sum((x - mean)**k for x in numbers) / n

def skewness(numbers):
    """ 
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = skewness(numbers)
    print(result)
    """
    n = len(numbers)
    mean = np.mean(numbers)
    std = np.std(numbers)
    return sum(((x - mean) / std)**3 for x in numbers) / n


