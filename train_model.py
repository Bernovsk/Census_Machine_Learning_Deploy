# Script to train machine learning model.
from ast import Constant
import pandas as pd
import constants
from ml.data import process_data, remove_dash, clean_spaces
from ml.model import *
from sklearn.model_selection import train_test_split


def load_data(path:str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep = ';')
    return frame


def data_preprocessing():
    data = load_data(path=constants.DATA_PATH)

    frame = data.copy()

    for coluna in frame.select_dtypes('object').columns.tolist():
        frame = remove_dash(frame = frame, col_name = coluna)
        frame = clean_spaces(frame = frame, col_name = coluna)
    
    frame.to_csv(f"{Constant.DATA_PATH}/pre_processing_census_data.csv")
    train_frame, test_frame = train_test_split(frame, test_size = 0.20, random_state = 42)
    return train_frame, test_frame


def model_preprocessing():
    train_frame, _ = data_preprocessing()
    X_train, y_train, encoder, lb = process_data(
        train_frame, categorical_features=constants.CATEGORICAL_FEATURES, label="salary", training=True
    )
    return X_train, y_train, encoder, lb


def train_model(path):
    
    X_train, y_train, encoder, lb = model_preprocessing()

    model = train_model(X_train, y_train)

    save_dump(model, constants.MODEL_PATH, "model")
    save_dump(encoder, constants.MODEL_PATH, "encoder")
    save_dump(lb, constants.MODEL_PATH, "lb")
    return model