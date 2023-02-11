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
- range Range of values
- mad  Mean or median absolute deviation
- zscore  Standardized z-scores

## Correlation and Covariance
corr  Linear or rank correlation
robustcov  Robust multivariate covariance and mean estimate
cholcov  Cholesky-like covariance decomposition
corrcov  Convert covariance matrix to correlation matrix
partialcorr  Linear or rank partial correlation coefficients
partialcorri  Partial correlation coefficients adjusted for internal variables
nearcorr   Compute nearest correlation matrix by minimizing Frobenius distance

## 
"""
import math 
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from scipy.linalg import inv
from scipy.stats import pearsonr
from scipy.optimize import minimize

__all__ = ['geomean','harmmean','trimean','kurtosis','moment','skewness','zscore','corr','robustcov',
          'cholcov','corrcov','partialcorr','partialcorri','nearcorr']

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


def moment(numbers, k):
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


def zscore(x,data):
    mean = sum(data) / len(data)
    stddev = (sum([(x - mean)**2 for x in data]) / len(data))**0.5
    z = (x - mean) / stddev
    return z

def corr(x,y,method):
    """
    mathod 
    ----------
    1. pearson
    2. spearman
    3. kendal
    """
    if(method == 'pearson'):
        return pearson_correlation(x,y)
    elif(method == 'spearman'):
        return spearman_correlation(x,y)
    elif(method == 'kendal'):
        return  kendall_correlation(x, y)

def pearson_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    cov = np.cov(x, y)[0][1]
    return cov / (x_std * y_std)

def spearman_correlation(x, y):
    x_rank = np.argsort(x)
    y_rank = np.argsort(y)
    x_d = np.array([x_rank[i] - y_rank[i] for i in range(len(x))])
    return 1 - (6 * np.sum(x_d**2)) / (len(x) * (len(x)**2 - 1))


def kendall_correlation(x, y):
    n = len(x)
    x_rank = sorted(range(len(x)), key=lambda i: x[i])
    y_rank = sorted(range(len(y)), key=lambda i: y[i])
    concordant_pairs = 0
    discordant_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            if (x_rank[i] < x_rank[j] and y_rank[i] < y_rank[j]) or (x_rank[i] > x_rank[j] and y_rank[i] > y_rank[j]):
                concordant_pairs += 1
            else:
                discordant_pairs += 1
    return (concordant_pairs - discordant_pairs) / (n * (n - 1) / 2)


def robustcov(X):
    mcd = MinCovDet().fit(X)
    return mcd.covariance_, mcd.location_


def cholcov(covariance_matrix):
    n = covariance_matrix.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(covariance_matrix[i, i] - s)
            else:
                L[i, j] = (covariance_matrix[i, j] - s) / L[j, j]
    return L

def corrcov(covariance_matrix):
    n = covariance_matrix.shape[0]
    correlation_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            correlation_matrix[i, j] = covariance_matrix[i, j] / np.sqrt(covariance_matrix[i, i] * covariance_matrix[j, j])
    return correlation_matrix


def partialcorr(data, var1, var2, controlling_vars):
    X = np.column_stack((data[:, controlling_vars], data[:, [var1, var2]]))
    cov = np.cov(X, rowvar=False)
    n = X.shape[1]
    partial_correlation = -cov[n - 2, n - 1] / np.sqrt(cov[n - 2, n - 2] * cov[n - 1, n - 1])
    return partial_correlation


def partialcorri(x, y, covars):
    """
    Calculate the partial correlation coefficient between two variables x and y, 
    adjusting for the effect of a set of covariates.
    
    Parameters:
        x, y (np.array): arrays of the two variables to be correlated
        covars (np.array): array of covariates
    
    Returns:
        tuple: Pearson's correlation coefficient and p-value
    """
    # Calculate the residuals for x and y after regressing out the effect of the covariates
    X = np.column_stack((covars, x))
    beta_x = np.linalg.lstsq(X, y, rcond=None)[0]
    res_x = y - X @ beta_x
    Y = np.column_stack((covars, y))
    beta_y = np.linalg.lstsq(Y, x, rcond=None)[0]
    res_y = x - Y @ beta_y
    
    # Calculate the partial correlation coefficient between the residuals
    corr, p_value = pearsonr(res_x, res_y)
    return corr, p_value


def nearcorr(matrix):
    """
    Compute the nearest correlation matrix for a given matrix by minimizing the Frobenius distance.
    
    Parameters:
        matrix (np.array): The input matrix to find the nearest correlation matrix for.
        
    Returns:
        np.array: The nearest correlation matrix.
    """
    n = matrix.shape[0]
    
    # Constraints for optimization
    bounds = [(1e-15, 1 - 1e-15) for i in range(n * (n - 1) // 2)]
    
    # Minimize the Frobenius distance between the input matrix and the nearest correlation matrix
    def frobenius_distance(x):
        k = 0
        nearest_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                nearest_matrix[i, j] = x[k]
                nearest_matrix[j, i] = x[k]
                k += 1
        nearest_matrix = (nearest_matrix + nearest_matrix.T) / 2
        nearest_matrix = nearest_matrix + np.eye(n)
        return np.linalg.norm(nearest_matrix - matrix, 'fro')
    
    result = minimize(frobenius_distance, [0.5 for i in range(n * (n - 1) // 2)], bounds=bounds)
    x = result.x
    k = 0
    nearest_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            nearest_matrix[i, j] = x[k]
            nearest_matrix[j, i] = x[k]
            k += 1
    nearest_matrix = (nearest_matrix + nearest_matrix.T) / 2
    nearest_matrix = nearest_matrix + np.eye(n)
    return nearest_matrix
