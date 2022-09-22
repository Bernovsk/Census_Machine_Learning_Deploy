"""
Module that train the model
Author: Bernardo C.
Date: 2022/09/21
"""
from typing import Union
import logging
from ml.model_ import train_model, save_dump
from ml.constants import MODEL_PATH, PROCESSED_CAT_FEATURES
from ml.data import data_preprocessing, train_test_data, process_data


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def model_preprocessing(data_path: Union[str, None] = None):
    """
    Model preprocessing for the train step

    Input:
        data_path: (str or None)
            data path that contains the
            raw data for the training step
    Output:
        X_train: (np.array)
            Training data
        y_train: (np.array)
            Target data (true label)
        encoder:
            model encoder
        lb:
            model lb
    """
    logging.info("Preprocessing the Data for the model")
    preprocessed_data = data_preprocessing(data_path)

    logging.info("Spliting the data into train test")
    train_frame, _ = train_test_data(preprocessed_data, save=True)

    logging.info("Processing the categorical and continuous data")
    train_data, target_data, encoder_model, lb_model = process_data(
        train_frame, categorical_features=PROCESSED_CAT_FEATURES, label="salary", training=True)
    return train_data, target_data, encoder_model, lb_model


def train_model_implementation(data_path: Union[str, None] = None):
    """
    Training the model

    Input:
        data_path: (str or None)
            data path that contains the
            raw data for the training step
    Output:
        model:
            Trained Model
    """
    train_data, target_data, encoder_model, lb_model = model_preprocessing(data_path)
    logging.info("Training the model")
    model = train_model(train_data, target_data)

    logging.info("Saving the model")
    save_dump(model, MODEL_PATH, "model")
    save_dump(encoder_model, MODEL_PATH, "encoder")
    save_dump(lb_model, MODEL_PATH, "lb")
    return model
