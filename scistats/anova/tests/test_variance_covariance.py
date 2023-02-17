from typing import List
from scipy.stats import f_oneway

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
