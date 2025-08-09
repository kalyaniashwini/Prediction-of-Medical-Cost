import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    """
    Loads the dataset and applies label encoding to categorical columns.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        data (pd.DataFrame): Preprocessed dataframe.
        label_encoders (dict): Dictionary of fitted label encoders.
    """
    data = pd.read_csv(filepath)
    label_encoders = {}
    for col in ['sex', 'smoker', 'region']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    return data, label_encoders
