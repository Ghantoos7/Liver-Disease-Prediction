import pandas as pd



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


import pandas as pd

def convert_gender_to_numeric(df, column_name='Gender'):
    """
    Converts the 'Gender' column of a DataFrame to numeric values.

    Parameters:
    - df: The DataFrame containing the 'Gender' column.
    - column_name: The name of the gender column to convert (default is 'Gender').

    Returns:
    - The DataFrame with the 'Gender' column converted to numeric values.
    """
    gender_mapping = {'Male': 1, 'Female': 0}
    if column_name in df.columns:
        df[column_name] = df[column_name].map(gender_mapping)
    else:
        print(f"Column '{column_name}' not found in DataFrame.")
    return df



    