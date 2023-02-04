""" 
Descriptive Statistics and Visualization 

Categories:
1.) Managing Data: Data import and export grouping variables
2.) Descriptive statistics : Numerical summarises and associated measures
3.) statistical visualizaiton : view data patterns and trens 


# Central Tendency and Dispersion
- geomean 
- harmmean 
- trimean 
- kurtosis
- moment
- skewness


"""
import math 

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

def trimean():
    pass

def kurtosis():
    pass

def moment():
    pass

def skewness():
    pass 



