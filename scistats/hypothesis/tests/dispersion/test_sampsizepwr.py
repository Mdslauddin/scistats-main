import numpy as np
from statsmodels.stats.power import tt_ind_solve_power

def calculate_sample_size(effect_size, alpha, power, ratio=1, alternative='two-sided'):
    """
    Calculates the required sample size for a two-sample t-test.
    
    Parameters:
    effect_size (float): The standardized effect size (Cohen's d).
    alpha (float): The significance level of the test.
    power (float): The desired power of the test.
    ratio (float): The ratio of the sample sizes of the two groups (default 1).
    alternative (str): The alternative hypothesis ('two-sided', 'larger', or 'smaller') (default 'two-sided').
    
    Returns:
    sample_size (int): The required sample size per group.
    """
    sample_size = tt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=ratio, alternative=alternative)
    return int(np.ceil(sample_size))

def calculate_power(sample_size, effect_size, alpha, ratio=1, alternative='two-sided'):
    """
    Calculates the power of a two-sample t-test with the given parameters.
    
    Parameters:
    sample_size (int): The sample size per group.
    effect_size (float): The standardized effect size (Cohen's d).
    alpha (float): The significance level of the test.
    ratio (float): The ratio of the sample sizes of the two groups (default 1).
    alternative (str): The alternative hypothesis ('two-sided', 'larger', or 'smaller') (default 'two-sided').
    
    Returns:
    power (float): The power of the test.
    """
    power = tt_ind_solve_power(effect_size=effect_size, nobs1=sample_size, alpha=alpha, ratio=ratio, alternative=alternative)
    return power
