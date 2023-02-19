# Repeated Measures and MANOVA

__all__ = ['fitrm', 'ranova', 'mauchly', 'epsilon', 'multcompare', 'anova', 'manova', 'coeftest', 'grpstats',
          'margmean']

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def fitrm(data, dependent_variable, time_variable, treatment_variable, subject_variable):
    """
    fit_repeated_measures
    Fits a repeated measures model to a given dataset.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data to be analyzed.
    dependent_variable (str): The name of the dependent variable.
    time_variable (str): The name of the time variable.
    treatment_variable (str): The name of the treatment variable.
    subject_variable (str): The name of the subject variable.

    Returns:
    model (statsmodels.regression.linear_model.RegressionResultsWrapper): A statsmodels regression model object.
    """
    # Fit a repeated measures model using the statsmodels library
    formula = f"{dependent_variable} ~ {time_variable} + {treatment_variable} + {time_variable}:{treatment_variable} + {subject_variable}"
    model = ols(formula, data=data).fit()

    # Print the model summary
    print(model.summary())

    return model



import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

def ranova(data, dependent_variable, subject_variable, within_variables):
    """
    repeated_measures_anova
    Performs a repeated measures ANOVA on a given dataset.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data to be analyzed.
    dependent_variable (str): The name of the dependent variable.
    subject_variable (str): The name of the subject variable.
    within_variables (list of str): A list of the names of the within-subject variables.

    Returns:
    model (statsmodels.stats.anova.anova.AnovaRM): A statsmodels ANOVA model object.
    """
    # Perform a repeated measures ANOVA using the statsmodels library
    model = AnovaRM(data, depvar=dependent_variable, subject=subject_variable, within=within_variables).fit()

    # Print the ANOVA table
    print(model.summary())

    return model


import pandas as pd
import scipy.stats as stats

def mauchly(data, within_variables):
    """
    mauchlys_test
    Performs Mauchly's test for sphericity on a given dataset.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data to be analyzed.
    within_variables (list of str): A list of the names of the within-subject variables.

    Returns:
    result (str): A string describing the result of the test.
    p_value (float): The p-value of the test.
    """
    # Extract the within-subject variables from the data
    within_data = data[within_variables]

    # Calculate the covariance matrix of the within-subject variables
    cov_matrix = within_data.cov()

    # Calculate the determinant of the covariance matrix
    det = np.linalg.det(cov_matrix)

    # Calculate the trace of the covariance matrix
    trace = np.trace(cov_matrix)

    # Calculate the number of subjects
    n = data[within_variables[0]].nunique()

    # Calculate the test statistic
    w = (det / ((trace ** 2) / ((n * (n + 1)) / 2)))

    # Calculate the p-value of the test
    df1 = (len(within_variables) * (n - 1))
    p_value = stats.chi2.sf(w, df1)

    # Determine the result of the test based on the p-value
    if p_value < 0.05:
        result = "The assumption of sphericity has been violated."
    else:
        result = "The assumption of sphericity has not been violated."

    return result, p_value


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

def epsilon(data, dependent_variable, subject_variable, within_variables):
    """
    repeated_measures_anova
    Performs a repeated measures ANOVA on a given dataset.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data to be analyzed.
    dependent_variable (str): The name of the dependent variable.
    subject_variable (str): The name of the subject variable.
    within_variables (list of str): A list of the names of the within-subject variables.

    Returns:
    model (statsmodels.stats.anova.anova.AnovaRM): A statsmodels ANOVA model object.
    """
    # Perform a repeated measures ANOVA using the statsmodels library
    model = AnovaRM(data, depvar=dependent_variable, subject=subject_variable, within=within_variables).fit()

    # Calculate epsilon using the Greenhouse-Geisser method
    eps = model.epsilon

    # Adjust the degrees of freedom and p-values using epsilon
    df = model.df.values * eps
    p_values = model.pvalues.values

    # Create a new ANOVA table with the adjusted degrees of freedom and p-values
    new_anova_table = pd.DataFrame({'SS': model.anova_table['SS'].values,
                                    'MS': model.anova_table['MS'].values,
                                    'df': df,
                                    'F': model.anova_table['F'].values,
                                    'PR(>F)': p_values},
                                    index=model.anova_table.index)

    # Print the adjusted ANOVA table
    print(new_anova_table)

    return model


import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison

def multcompare(data, dependent_variable, between_variables, alpha=0.05):
    """
    emm_multiple_comparison
    Performs multiple comparisons of estimated marginal means using Tukey's HSD test.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data to be analyzed.
    dependent_variable (str): The name of the dependent variable.
    between_variables (list of str): A list of the names of the between-subject variables.
    alpha (float): The significance level for the Tukey's HSD test.

    Returns:
    results (statsmodels.sandbox.stats.multicomp.TukeyHSDResults): A statsmodels TukeyHSDResults object.
    """
    # Create a formula for the EMMs using the statsmodels formula API
    formula = f"{dependent_variable} ~ {'*'.join(between_variables)}"

    # Fit a generalized linear model using the formula
    model = sm.formula.glm(formula, data=data).fit()

    # Calculate the EMMs using the model and the between-subject variables
    emm = model.get_margeff(at='overall', method='dydx')

    # Perform multiple comparisons of the EMMs using Tukey's HSD test
    mc = MultiComparison(emm['mean'], emm.index)
    results = mc.tukeyhsd(alpha=alpha)

    # Print the results of the multiple comparison test
    print(results.summary())

    return results


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def anvoa(data, dependent_variable, between_variables):
    """
    between_subjects_anova
    Performs an analysis of variance (ANOVA) for between-subject effects.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data to be analyzed.
    dependent_variable (str): The name of the dependent variable.
    between_variables (list of str): A list of the names of the between-subject variables.

    Returns:
    results (statsmodels.anova.anova.AnovaRM): A statsmodels AnovaRM object.
    """
    # Create a formula for the ANOVA using the statsmodels formula API
    formula = f"{dependent_variable} ~ {'+'.join(between_variables)}"

    # Fit a linear model using the formula
    model = ols(formula, data=data).fit()

    # Perform an ANOVA for between-subject effects using the linear model
    results = anova_lm(model, typ=2)

    # Print the results of the ANOVA
    print(results)

    return results


import numpy as np
import pandas as pd
import statsmodels.api as sm

def manova(data, dependent_variables, independent_variable):
    """
    Performs a multivariate analysis of variance (MANOVA).

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data to be analyzed.
    dependent_variables (list of str): A list of the names of the dependent variables.
    independent_variable (str): The name of the independent variable.

    Returns:
    results (statsmodels.multivariate.manova.Manova): A statsmodels Manova object.
    """
    # Create a design matrix with the dependent variables and the independent variable
    design = data[dependent_variables + [independent_variable]].to_numpy()

    # Create a list of column names for the design matrix
    column_names = dependent_variables + [independent_variable]

    # Center the dependent variables
    centered_data = data[dependent_variables] - data[dependent_variables].mean()

    # Fit a multivariate linear model using the centered data and the independent variable
    model = sm.multivariate.MANOVA(endog=centered_data, exog=sm.add_constant(data[independent_variable])).fit()

    # Print the results of the MANOVA
    print(model.summary())

    
    
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.contrast import LinearContrast

def coeftest(data, formula, hypothesis, cov_type='nonrobust'):
    """
    repeated_measures_linear_hypothesis
    Performs a linear hypothesis test on the coefficients of a repeated measures model.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data to be analyzed.
    formula (str): A string specifying the formula for the repeated measures model.
    hypothesis (array-like): A list or array specifying the linear hypothesis to be tested.
    cov_type (str): The type of covariance estimator to be used.

    Returns:
    results (statsmodels.regression.linear_model.RegressionResults): A statsmodels RegressionResults object.
    """
    # Fit the repeated measures model using OLS
    model = OLS.from_formula(formula, data=data).fit(cov_type=cov_type)

    # Perform the linear hypothesis test using the LinearContrast class
    contrast = LinearContrast(model.params, hypothesis)
    results = contrast.t_test()

    # Print the results of the hypothesis test
    print(results)

    return results

import numpy as np
import pandas as pd

def grpstats(data, group_col, time_col, value_col):
    """
    repeated_measures_descriptive_stats
    Computes descriptive statistics of repeated measures data by group.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the repeated measures data.
    group_col (str): The name of the column containing the group labels.
    time_col (str): The name of the column containing the time variable.
    value_col (str): The name of the column containing the measured values.

    Returns:
    stats (pandas DataFrame): A pandas DataFrame containing the descriptive statistics for each group.
    """
    # Compute the mean, standard deviation, standard error, and sample size for each group at each time point
    stats = data.groupby([group_col, time_col])[value_col].agg(['mean', 'std', 'sem', 'count']).reset_index()

    # Pivot the data to put each time point in a separate column
    stats = stats.pivot(index=group_col, columns=time_col)

    # Rename the columns to include the time point and the statistic name
    new_cols = [f'{time}_{stat}' for time, stat in stats.columns]
    stats.columns = new_cols

    # Flatten the multi-level column index
    stats = stats.reset_index()
    stats.columns = stats.columns.map('_'.join)

    return stats


import numpy as np
import pandas as pd
import statsmodels.api as sm

def margmean(data, model_formula, groups, covariates=None):
    """
    estimate_marginal_means
    Estimates marginal means for each level of the grouping variable(s) using a linear model.

    Args:
    data (pandas DataFrame): A pandas DataFrame containing the data.
    model_formula (str): A string representing the formula for the linear model.
    groups (list of str): A list of column names representing the grouping variable(s).
    covariates (list of str, optional): A list of column names representing the covariates.

    Returns:
    marginal_means (pandas DataFrame): A pandas DataFrame containing the estimated marginal means for each level of the grouping variable(s).
    """
    # Fit the linear model
    if covariates is None:
        model = sm.formula.ols(model_formula, data).fit()
    else:
        model_formula = f'{model_formula} + {" + ".join(covariates)}'
        model = sm.formula.ols(model_formula, data).fit()

    # Create a new DataFrame with the combinations of group levels
    group_values = [data[group].unique() for group in groups]
    group_combinations = np.array(np.meshgrid(*group_values)).T.reshape(-1, len(groups))
    group_df = pd.DataFrame(group_combinations, columns=groups)

    # Predict the response variable for each group combination
    predictions = model.predict(group_df)

    # Add the predicted values to the group DataFrame
    group_df['predicted'] = predictions

    # Compute the marginal means for each group level
    marginal_means = group_df.groupby(groups).mean().reset_index()

    return marginal_means
