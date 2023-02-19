from typing import List
from scipy.stats import f_oneway


__all__ = ['anova', 'anova1', 'anova2', 'anovan', 'aoctool', 'cannoncorr', 'dummyvar', 'friedman', 'kruskalwallis',
           'multcompare']


def perform_anova(groups: List[List[float]], significance_level: float = 0.05) -> tuple:
    """
    Perform an ANOVA test on a list of groups with sample data.

    Parameters:
        - groups: a list of lists of float values, where each inner list represents a group of sample data
        - significance_level: the significance level for the test (default: 0.05)

    Returns:
        - A tuple containing the F statistic and corresponding p-value from the ANOVA test.
    """

    # Perform ANOVA test
    f_statistic, p_value = f_oneway(*groups)

    # Print ANOVA results
    print("F statistic:", f_statistic)
    print("p-value:", p_value)

    # Check significance level
    if p_value < significance_level:
        print("Reject null hypothesis: At least one group mean is significantly different from the others.")
    else:
        print("Failed to reject null hypothesis: There is no significant difference between group means.")

    return f_statistic, p_value


def anova1(data):
    """
    one_way_anova
    Performs one-way analysis of variance (ANOVA) on a given dataset.

    Args:
    data (list of lists): A list of lists, where each sublist contains the data for one group.

    Returns:
    F (float): The F-statistic.
    p_value (float): The p-value.
    """
    # Compute the overall mean and the number of groups and observations
    n_groups = len(data)
    n_obs = sum([len(x) for x in data])
    overall_mean = sum([sum(x) for x in data]) / n_obs

    # Compute the between-group sum of squares (SSB) and the degrees of freedom (DFB)
    SSB = sum([len(x) * ((sum(x) / len(x)) - overall_mean) ** 2 for x in data])
    DFB = n_groups - 1

    # Compute the within-group sum of squares (SSW) and the degrees of freedom (DFW)
    SSW = sum([sum([(x - (sum(y) / len(y))) ** 2 for x in y]) for y in data])
    DFW = n_obs - n_groups

    # Compute the F-statistic and the p-value
    F = (SSB / DFB) / (SSW / DFW)
    p_value = 1 - scipy.stats.f.cdf(F, DFB, DFW)

    return F, p_value


def anova2(data, alpha=0.05):
    """
    two_way_anova
    
    Performs two-way analysis of variance (ANOVA) on a given dataset.

    Args:
    data (list of lists of lists): A list of lists of lists, where the outer list contains the data for each level of
    the first factor, the middle list contains the data for each level of the second factor, and the inner list contains
    the data for each observation.
    alpha (float): The significance level for the F-tests.

    Returns:
    F1 (float): The F-statistic for the first factor.
    p1 (float): The p-value for the first factor.
    F2 (float): The F-statistic for the second factor.
    p2 (float): The p-value for the second factor.
    F_interaction (float): The F-statistic for the interaction effect.
    p_interaction (float): The p-value for the interaction effect.
    """
    # Compute the overall mean and the number of levels and observations for each factor
    n_levels_1 = len(data)
    n_levels_2 = len(data[0])
    n_obs = sum([len(x) for y in data for x in y])
    overall_mean = sum([sum([sum(x) for x in y]) for y in data]) / n_obs

    # Compute the sum of squares (SS) and degrees of freedom (DF) for each factor and the interaction effect
    SS_1 = sum([len(y) * ((sum([sum(x) for x in y]) / (len(y) * n_levels_2)) - overall_mean) ** 2 for y in data])
    DF_1 = n_levels_1 - 1
    SS_2 = sum([len(x) * ((sum([sum(y[i]) for y in data]) / (len(x) * n_levels_1)) - overall_mean) ** 2 for i, x in enumerate(data[0])])
    DF_2 = n_levels_2 - 1
    SS_interaction = sum([(sum([sum(x) for x in y]) ** 2 / (len(y) * n_levels_2)) - (sum([sum(x) for x in y]) / len(y)) ** 2 for y in data])
    DF_interaction = (n_levels_1 - 1) * (n_levels_2 - 1)

    # Compute the error sum of squares (SSE) and degrees of freedom (DFE)
    SSE = sum([sum([(x - (sum([sum(z) for z in y]) / (len(y) * n_levels_2)) ** 2) for x in y]) for y in data])
    DFE = n_obs - n_levels_1 * n_levels_2

    # Compute the F-statistics and p-values for each factor and the interaction effect
    F1 = (SS_1 / DF_1) / (SSE / DFE)
    p1 = 1 - scipy.stats.f.cdf(F1, DF_1, DFE)
    F2 = (SS_2 / DF_2) / (SSE / DFE)
    p2 = 1 - scipy.stats.f.cdf(F2, DF_2, DFE)
    F_interaction = (SS_interaction / DF_interaction) / (SSE / DFE)
    p_interaction = 1 - scipy.stats.f.cdf(F_interaction, DF_interaction, DFE)

    # Perform significance tests for each factor and the interaction effect
"""
import numpy as np
import scipy.stats

def two_way_anova(data, alpha):
    """
    Performs two-way analysis of variance (ANOVA) on a given dataset.

    Args:
    data (numpy array): A 2D numpy array with n rows and 3 columns. 
                        The first column should contain the response variable, 
                        and the remaining two columns should contain the two 
                        factors (categorical variables) to be analyzed.
    alpha (float): The significance level of the test.

    Returns:
    F1 (float): The F-statistic for the first factor.
    p1 (float): The p-value for the first factor.
    F2 (float): The F-statistic for the second factor.
    p2 (float): The p-value for the second factor.
    F_interaction (float): The F-statistic for the interaction between the two factors.
    p_interaction (float): The p-value for the interaction between the two factors.
    """
    # Calculate the overall mean
    overall_mean = np.mean(data[:, 0])

    # Calculate the sum of squares for factor 1
    ss_factor1 = sum((np.mean(data[data[:, 1] == i, 0]) - overall_mean)**2 for i in set(data[:, 1]))
    df_factor1 = len(set(data[:, 1])) - 1

    # Calculate the sum of squares for factor 2
    ss_factor2 = sum((np.mean(data[data[:, 2] == i, 0]) - overall_mean)**2 for i in set(data[:, 2]))
    df_factor2 = len(set(data[:, 2])) - 1

    # Calculate the sum of squares for the interaction between factor 1 and factor 2
    ss_interaction = 0
    for i in set(data[:, 1]):
        for j in set(data[:, 2]):
            ss_interaction += (np.mean(data[(data[:, 1] == i) & (data[:, 2] == j), 0]) - 
                               np.mean(data[data[:, 1] == i, 0]) -
                               np.mean(data[data[:, 2] == j, 0]) +
                               overall_mean)**2
    df_interaction = (len(set(data[:, 1])) - 1) * (len(set(data[:, 2])) - 1)

    # Calculate the sum of squares for error
    ss_error = sum((data[data[:, 1] == i, 0] - np.mean(data[data[:, 1] == i, 0]) -
                    data[data[:, 2] == j, 0] + np.mean(data[data[:, 2] == j, 0]) -
                    overall_mean + np.mean(data[(data[:, 1] == i) & (data[:, 2] == j), 0]) -
                    data[(data[:, 1] == i) & (data[:, 2] == j), 0])**2 for i in set(data[:, 1]) for j in set(data[:, 2]))
    df_error = (len(set(data[:, 1])) * len(set(data[:, 2]))) - len(data)

    # Calculate the mean squares and F-statistics for each factor and the interaction
    ms_factor1 = ss_factor1 / df_factor1
    ms_factor2 = ss_factor2 / df_factor2
    ms_interaction = ss_interaction / df_interaction
    ms_error = ss_error / df_error

    F1 = ms_factor1 / ms_error
    F2 = ms_factor2


"""
    
    
import numpy as np
import scipy.stats

def anovan(data, factors, alpha):
    """
    n_way_anova
    
    Performs N-way analysis of variance (ANOVA) on a given dataset.

    Args:
    data (numpy array): A 2D numpy array with n rows and m+1 columns, 
                        where n is the number of observations and m is the number of factors.
                        The first column should contain the response variable, 
                        and the remaining m columns should contain the m 
                        factors (categorical variables) to be analyzed.
    factors (list): A list of integers representing the number of levels for each factor.
    alpha (float): The significance level of the test.

    Returns:
    F_values (list): A list of F-statistics for each factor.
    p_values (list): A list of p-values for each factor.
    """
    # Calculate the overall mean
    overall_mean = np.mean(data[:, 0])

    # Calculate the sum of squares and degrees of freedom for each factor
    ss = []
    df = []
    for i in range(len(factors)):
        levels = factors[i]
        ss_factor = 0
        for level in range(levels):
            # Calculate the mean for the current level of the factor
            mean = np.mean(data[data[:, i+1] == level, 0])
            # Calculate the sum of squares for the current level of the factor
            ss_factor += (mean - overall_mean)**2 * sum(data[:, i+1] == level)
        df_factor = levels - 1
        ss.append(ss_factor)
        df.append(df_factor)

    # Calculate the sum of squares for the interaction between each pair of factors
    for i in range(len(factors)-1):
        for j in range(i+1, len(factors)):
            levels_i = factors[i]
            levels_j = factors[j]
            ss_interaction = 0
            for level_i in range(levels_i):
                for level_j in range(levels_j):
                    # Calculate the mean for the current combination of levels of the two factors
                    mean = np.mean(data[(data[:, i+1] == level_i) & (data[:, j+1] == level_j), 0])
                    # Calculate the sum of squares for the current combination of levels of the two factors
                    ss_interaction += (mean - np.mean(data[data[:, i+1] == level_i, 0]) -
                                       np.mean(data[data[:, j+1] == level_j, 0]) +
                                       overall_mean)**2 * sum((data[:, i+1] == level_i) & (data[:, j+1] == level_j))
            df_interaction = (levels_i - 1) * (levels_j - 1)
            ss.append(ss_interaction)
            df.append(df_interaction)

    # Calculate the sum of squares for error
    ss_error = 0
    for i in range(len(factors)):
        levels = factors[i]
        for level in range(levels):
            # Calculate the mean for the current level of the factor
            mean = np.mean(data[data[:, i+1] == level, 0])
            # Calculate the sum of squares for the current level of the factor
            ss_error += sum((data[:, i+1] == level) * (data[:, 0] - mean)**2)
    df_error = data.shape[0] - sum(df) - 1

    # Calculate the mean squares and F-statistics for each factor and the interaction
    ms = [ss[i] / df[i] for

          
import pandas as pd
import statsmodels.formula.api as smf

def aoctool(data, x_var, y_var, covar, group_var, alpha):
    """
    interactive_ancova
    Performs an interactive analysis of covariance (ANCOVA) on a given dataset.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data to be analyzed.
    x_var (str): The name of the independent variable (a continuous variable).
    y_var (str): The name of the dependent variable (a continuous variable).
    covar (str): The name of the covariate (a continuous variable).
    group_var (str): The name of the grouping variable (a categorical variable).
    alpha (float): The significance level of the test.

    Returns:
    results (statsmodels results object): The results of the ANCOVA.
    """
    # Fit the ANCOVA model using the statsmodels library
    formula = "{} ~ {} + {} + {} + {}*{}".format(y_var, x_var, covar, group_var, x_var, group_var)
    model = smf.ols(formula, data=data).fit()

    # Print the summary of the model
    print(model.summary())

    # Test the main effect of the independent variable
    results = model.f_test("{} = 0".format(x_var))

    # Print the results of the test
    print("Main effect of {}: F = {:.3f}, p-value = {:.4f}".format(x_var, results.fvalue[0][0], results.pvalue))

    # Test the main effect of the grouping variable
    results = model.f_test("{} = 0".format(group_var))

    # Print the results of the test
    print("Main effect of {}: F = {:.3f}, p-value = {:.4f}".format(group_var, results.fvalue[0][0], results.pvalue))

    # Test the interaction effect between the independent variable and the grouping variable
    results = model.f_test("{}:{} = 0".format(x_var, group_var))

    # Print the results of the test
    print("Interaction effect between {} and {}: F = {:.3f}, p-value = {:.4f}".format(x_var, group_var, results.fvalue[0][0], results.pvalue))

    return model

          
          
          
from sklearn.cross_decomposition import CCA
import numpy as np

def cannoncorr(data, x_vars, y_vars, num_cc):
    """
    canonical_correlation
    Performs canonical correlation analysis on a given dataset.

    Args:
    data (numpy array): A numpy array containing the data to be analyzed.
    x_vars (int list): The indices of the columns containing the predictor variables.
    y_vars (int list): The indices of the columns containing the response variables.
    num_cc (int): The number of canonical correlation pairs to compute.

    Returns:
    cc (numpy array): The canonical correlation coefficients.
    x_weights (numpy array): The canonical correlation weights for the predictor variables.
    y_weights (numpy array): The canonical correlation weights for the response variables.
    """
    # Split the data into predictor and response variables
    x = data[:, x_vars]
    y = data[:, y_vars]

    # Fit the CCA model using the sklearn library
    cca = CCA(n_components=num_cc)
    cca.fit(x, y)

    # Get the canonical correlation coefficients, weights, and scores
    cc = cca.correlation_
    x_weights = cca.x_weights_
    y_weights = cca.y_weights_

    # Print the canonical correlations and weights
    print("Canonical Correlations: {}".format(cc))
    print("X weights: {}".format(x_weights))
    print("Y weights: {}".format(y_weights))

    return cc, x_weights, y_weights

          
import pandas as pd

def dummyvar(data, var_name):
    """
    create_dummies
    Creates dummy variables from a categorical variable in a pandas DataFrame.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data to be analyzed.
    var_name (str): The name of the categorical variable.

    Returns:
    dummies (pandas DataFrame): A pandas DataFrame containing the dummy variables.
    """
    # Create the dummy variables using the pandas get_dummies function
    dummies = pd.get_dummies(data[var_name], prefix=var_name)

    # Drop the original categorical variable from the DataFrame
    data = data.drop(var_name, axis=1)

    # Concatenate the original DataFrame with the dummy variable DataFrame
    data = pd.concat([data, dummies], axis=1)

    return data

          
     
from scipy.stats import friedmanchisquare
import numpy as np

def friedman(data):
    """
    friedmans_test
    Performs Friedman's test on a given dataset.

    Args:
    data (numpy array): A numpy array containing the data to be analyzed.

    Returns:
    chi2 (float): The chi-squared statistic.
    p_value (float): The p-value associated with the chi-squared statistic.
    """
    # Compute the ranks for each row of the data
    ranks = np.apply_along_axis(lambda x: rankdata(x, method='average'), 1, data)

    # Compute the Friedman's test statistic and p-value using the scipy friedmanchisquare function
    chi2, p_value = friedmanchisquare(*ranks)

    # Print the test results
    print("Friedman's Test:")
    print("Chi-Squared Statistic: {}".format(chi2))
    print("P-value: {}".format(p_value))

    return chi2, p_value
          
 
          
          
from scipy.stats import kruskal

def kruskalwallis(data):
    """
    kruskal_wallis
    Performs the Kruskal-Wallis test on a given dataset.

    Args:
    data (list of numpy arrays): A list of numpy arrays containing the data to be analyzed.

    Returns:
    H (float): The test statistic.
    p_value (float): The p-value associated with the test statistic.
    """
    # Perform the Kruskal-Wallis test using the scipy kruskal function
    H, p_value = kruskal(*data)

    # Print the test results
    print("Kruskal-Wallis Test:")
    print("Test Statistic: {}".format(H))
    print("P-value: {}".format(p_value))

    return H, p_value

          
   
import statsmodels.stats.multicomp as mc

def multcompare(data, groups):
    """
    
    tukey_hsd
    Performs the Tukey HSD test on a given dataset and set of groups.

    Args:
    data (numpy array): A numpy array containing the data to be analyzed.
    groups (list): A list of group labels for the data.

    Returns:
    results (pandas DataFrame): A pandas DataFrame containing the results of the test.
    """
    # Create a Tukey HSD object using the statsmodels library
    tukey = mc.MultiComparison(data, groups)

    # Perform the Tukey HSD test using the tukey object
    results = tukey.tukeyhsd()

    # Print the test results
    print("Tukey HSD Test:")
    print(results)

    return results.summary()
          