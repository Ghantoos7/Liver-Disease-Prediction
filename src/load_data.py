import openml
import pandas as pd

def download_openml_dataset(dataset_id):
    """
    Download a dataset from OpenML and return it as a pandas DataFrame.
    Parameters:
    - dataset_id: The ID of the dataset to be downloaded.

    Returns:
    - A DataFrame containing the downloaded dataset.
    """
    dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
    X, y, attribute_names, _ = dataset.get_data(target='Class', dataset_format='dataframe')
    df = pd.concat([X, y], axis=1)

    return df

