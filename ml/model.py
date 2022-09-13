import logging
from constants import *
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from joblib import dump





def save_dump(variable, path, response):
    """Save a variable into a joblib file
    
    Inputs:
    -------
    variable:
        Variable to be saved
    path:
        (str) Path to save the variable
    response:
        (str) Filename
    
    Output:
    -------
        NONE
    """
    dump(variable, f"{path}/model/{response}.joblib")


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    model = RandomForestClassifier(n_estimators = 100)
    model.fit(X_train, y_train)
    scores  = cross_val_score(model,
                             X_train,
                              y_train,
                               scoring = "accuracy",
                                cv = cv,
                                n_jobs = -1)
    logging.info(f"Accuracy: {round(np.mean(scores), 3)} ({round(np.std(scores), 3)})")
    return model


def compute_model_metrics(y, preds):
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
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
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
    predictions =  model.predict(X)

    return predictions


def slice_performance(model,
                      X_data: pd.DataFrame,
                      y_data,
                      selected_column : str,
                      helded_value : str):
    """ Run model infference over each slice of the categorical columns.
    Inputs
    ------
    model:
         Trained machine learning model.
    
    X_data:
        (np.array | pd.DataFrame) data used to train the model
    
    y_data:
        (np.array) the real target variable for metrics
    
    selected_column:
        (str) Selected sliced column
    
    helded_value:
        (str) Value to slice the selected Column
    
    Output
    ------
    valued_metrics:
        (str) String with the resulted metrics of the slice
    
    """
    frame = X_data.loc[X_data[selected_column] == helded_value].copy()
    input_frame = process_data(frame)

    predictions = inference(model, input_frame)
    precision, recall, fbeta = compute_model_metrics(y = y_data, preds = predictions)
    
    valued_metrics = f"""
                Column: {selected_column}
                Value Name: {helded_value}
                Precision: {round(precision, 2)} - Recall {round(recall, 2)} - Fbeta {round(fbeta, 3)}
                -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
                """

    logging.info(valued_metrics)
    return valued_metrics


def check_slices(model, X_data, y_data):
    with open(f"{MODEL_PATH}/sliced_output.txt", 'w') as output_file:
        for category in CATEGORICAL_FEATURES:
            unique_values = X_data[category].unique().tolist()
            for value in unique_values:
                output_string = slice_performance(model, X_data, y_data, category, value)
                output_file.write(output_string + '\n')