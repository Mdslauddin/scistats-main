import pandas as pd

def cross_tabulation(data, row_var, col_var):
    """
    Performs a cross-tabulation (contingency table) on a given dataset.

    Parameters:
        data (DataFrame): Pandas DataFrame containing the data to be analyzed.
        row_var (str): Name of the column to use as the row variable.
        col_var (str): Name of the column to use as the column variable.

    Returns:
        DataFrame: A Pandas DataFrame representing the contingency table.
    """
    cross_tab = pd.crosstab(data[row_var], data[col_var])
    return cross_tab
