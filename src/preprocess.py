



def rename_columns(df, column_names):

    """
    Renames the columns of a DataFrame with descriptive names.

    Parameters:
    - dataframe: The DataFrame whose columns are to be renamed.

    Returns:
    - A DataFrame with renamed columns.
    """

    df.columns = column_names
    return df



    