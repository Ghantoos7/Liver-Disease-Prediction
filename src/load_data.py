import openml
import pandas as pd
from scipy.io import arff



def download_openml_dataset(dataset_id):

    dataset = openml.datasets.get_dataset(dataset_id)
    data, _, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    if isinstance(data, pd.DataFrame):
        df = data
    else:
        data_arff = arff.loadarff(data)
        df = pd.DataFrame(data_arff[0])
    
    return df
