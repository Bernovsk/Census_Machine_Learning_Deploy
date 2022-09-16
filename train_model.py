from ml.model_ import *
from ml.constants import *
from ml.data import data_preprocessing, train_test_data
from typing import Union


import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def model_preprocessing(data_path: Union[str, None] = None):
    logging.info("Preprocessing the Data for the model")
    preprocessed_data = data_preprocessing(data_path = data_path)

    logging.info("Spliting the data into train test")
    train_frame, test_frame = train_test_data(preprocessed_data, save = True)

    logging.info("Processing the categorical and continuous data")
    X_train, y_train, encoder, lb = process_data(
        train_frame, categorical_features = PROCESSED_CAT_FEATURES, label="salary", training=True)
    return X_train, y_train, encoder, lb



def train_model_implementation(data_path: Union[str, None] = None):
    
    X_train, y_train, encoder, lb = model_preprocessing(data_path = data_path)
    logging.info("Training the model")
    model = train_model(X_train, y_train)

    logging.info("Saving the model")
    save_dump(model, MODEL_PATH, "model")
    save_dump(encoder, MODEL_PATH, "encoder")
    save_dump(lb, MODEL_PATH, "lb")
    return model