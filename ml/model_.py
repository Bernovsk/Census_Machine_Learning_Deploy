"""
This module contain the base functions
to train and test the model on slices

Author: Bernardo C.
Date: 2022/09/22
"""


import logging
from typing import Union
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from ml.constants import PROCESSED_CAT_FEATURES
from ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def save_dump(variable,
              path : str,
              response : str):
    """Save a variable into a joblib file

    Inputs:
    -------
    variable:
        Variable to be saved
    path: str
          Path to save the variable
    response: str
              Filename

    Output:
    -------
        None
    """
    try:
        logging.info("Saving the %s", str(response))
        dump(variable, f"{path}{response}.joblib")
    except:
        logging.error("rror on saving the %s", str(response))


def train_model(train_data: Union[np.array, pd.DataFrame],
                target_data: Union[np.array, pd.Series]):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    train_data : np.array
        Training data.
    target_data : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    crossval = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    model = RandomForestClassifier(n_estimators = 100)
    model.fit(train_data, target_data)
    scores  = cross_val_score(model,
                             train_data,
                              target_data,
                               scoring = "accuracy",
                                cv = crossval,
                                n_jobs = -1)
    logging.info("Accuracy: %s (%s)", round(np.mean(scores), 3), round(np.std(scores), 3))
    return model


def compute_model_metrics(true_target, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    print(len(true_target), len(preds))
    fbeta = fbeta_score(true_target, preds, beta=1, zero_division=1)
    precision = precision_score(true_target, preds, zero_division=1)
    recall = recall_score(true_target, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, inference_data):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    predictions : np.array
        Predictions from the model.
    """
    predictions =  model.predict(inference_data)

    return predictions


def slice_performance(model,
                      input_data: pd.DataFrame,
                      target_data,
                      selected_column : str,
                      helded_value : str):
    """ Run model infference over each slice of the categorical columns.
    Inputs
    ------
    model:
         Trained machine learning model.

    input_data:
        (np.array | pd.DataFrame) data used to test the model over the slices

    target_data:
        (np.array) the real label data for the input_data

    selected_column:
        (str) Selected sliced column

    helded_value:
        (str) Value to slice the selected Column

    Output
    ------
    valued_metrics:
        (str) String with the resulted metrics of the slice

    """
    predictions = inference(model, input_data)
    print('Checkpoint')
    precision, recall, fbeta = compute_model_metrics(true_target = target_data, preds = predictions)

    valued_metrics = f"""
                Column: {selected_column}
                Value Name: {helded_value}
                Precision: {round(precision, 2)} - Recall {round(recall, 2)} - Fbeta {round(fbeta, 3)}
                -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

                """
    logging.info(valued_metrics)
    return valued_metrics


def check_slices(model, ingested_data, encoder_model, lb_model):
    """
    Run the slice_performance function over each
    categorical column checking every unique value
    Input:
    ------

    """
    with open("./ml/model/sliced_output.txt", 'w', encoding='UTF-8') as output_file:
        for category in PROCESSED_CAT_FEATURES:
            unique_values = ingested_data[category].unique().tolist()
            for value in unique_values:
                sliced_frame = ingested_data.loc[ingested_data[category] == value].copy()
                sliced_frame.drop_duplicates(inplace = True)

                test_data, true_label, _, _ = process_data(X = sliced_frame,
                                                    categorical_features = PROCESSED_CAT_FEATURES,
                                                    lb = lb_model,
                                                    label = 'salary',
                                                    training=False,
                                                    encoder = encoder_model)

                output_string = slice_performance(model = model,
                                                  input_data = test_data,
                                                  target_data = true_label,
                                                  selected_column = category,
                                                  helded_value = value)
                print(output_string)
                output_file.write(output_string)

                del sliced_frame
