import pandas as pd
from scipy import stats
import numpy as np



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


from scipy import stats

def remove_outliers_z_score(df, threshold=3):
    """
    Removes rows containing outliers based on the Z-score method and returns the number of outliers.

    Parameters:
    - df: DataFrame to process, should contain only numerical columns for Z-score calculation.
    - threshold: Z-score value to use as the threshold for defining an outlier.

    Returns:
    - DataFrame with outliers removed.
    - Number of outliers removed.
    """
    z_scores = stats.zscore(df.select_dtypes(include=[float, int]))
    abs_z_scores = np.abs(z_scores)

    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    num_outliers = df.shape[0] - sum(filtered_entries)

    cleaned_df = df[filtered_entries]

    return cleaned_df, num_outliers




    