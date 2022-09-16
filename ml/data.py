import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from ml.constants import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder



def load_data(path:str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep = ',')
    frame_copy = frame.copy()
    frame_copy.columns = [col.strip().replace('-', '_') for col in frame_copy.columns]
    return frame_copy


def remove_dash(frame:pd.DataFrame, col_name:str) -> pd.DataFrame:
    """
    Remove the '-' char from any text column

    Input:
        frame: (pd.DataFrame)
        col_name: (str) Column that will replace the char value
    Ouput:
        frame_copy: (pd.DataFrame) replaced frame
    """
    frame_copy = frame.copy()
    frame_copy[col_name] = frame_copy[col_name].str.replace('-', '_')

    return frame_copy


def clean_spaces(frame:pd.DataFrame, col_name:str) -> pd.DataFrame:
    """
    Remove empty spaces from strings

    Input:
        frame: (pd.DataFrame)
        col_name: (str) Column that will remove the empty space
    Ouput:
        frame_copy: (pd.DataFrame) replaced frame
    """
    
    frame_copy = frame.copy()
    frame_copy[col_name] = frame_copy[col_name].apply(lambda x: str(x).strip())
    return frame_copy

def remove_unwanted(frame:pd.DataFrame, char:str = '?'):
    """
    Remove invalid values

    Input:
        frame: (pd.DataFrame)
        char: (str)
         String to removed
    Ouput:
        frame_copy: (pd.DataFrame)
         replaced frame
    """

    base_frame = frame.copy()
    for column in base_frame.columns.tolist():
        if column in base_frame.select_dtypes('object').columns.tolist():
            base_frame = base_frame.loc[base_frame[column] != char]

    base_frame.drop_duplicates(inplace = True)
    
    return base_frame

def data_preprocessing(data_path: Union[str, None] = None):

    if data_path == None:
        data = load_data(path= DATA_PATH + "census.csv")

    else:
        data = load_data(path=data_path)
    print('Checkpoint')
    frame = data.copy()

    for coluna in frame.columns.tolist():
        if coluna in frame.select_dtypes('object').columns.tolist():
            frame = clean_spaces(frame = frame, col_name = coluna)
            frame = remove_dash(frame = frame, col_name = coluna)
        clean_frame = remove_unwanted(frame)
    clean_frame.to_csv(f"{DATA_PATH}pre_processing_census_data.csv", index = False)
    return clean_frame


def train_test_data(clean_frame:pd.DataFrame, save : Union[bool, None]= False):

    clean_frame.to_csv(f"{DATA_PATH}pre_processing_census_data.csv", index = False)
    train_frame, test_frame = train_test_split(clean_frame, test_size = 0.20, random_state = 42)
    
    if save:
        train_frame.to_csv(f"{DATA_PATH}train_census_data.csv", index = False)
        test_frame.to_csv(f"{DATA_PATH}test_census_data.csv", index = False)
        
    return train_frame, test_frame



def process_data(
    X: Union[pd.DataFrame, np.array],
    categorical_features : Union [List[str], list] = [], 
    label : str = None,
    training : bool = True, 
    encoder: OneHotEncoder = None,
    lb : LabelBinarizer = None
    ) -> Tuple[np.array, np.array, OneHotEncoder, LabelBinarizer]:
    
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            print('y is none')
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb