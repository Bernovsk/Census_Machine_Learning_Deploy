"""
Module that Predict an output based on a external input

Author: Bernardo C.
Date: 2022/09/21
"""
import joblib as jb
from ml.model_ import inference
from ml.data import data_preprocessing, process_data
from ml.constants import PROCESSED_CAT_FEATURES


def run_inference(data):
    """
    Predicts the output of the model based on a raw data

    Input:
        data:(pd.DataFrame or Dictionary)
            Input data
    Output:
        (Str)
            Prediction of the model
    """
    model = jb.load("./ml/model/model.joblib")
    encoder_model = jb.load("./ml/model/encoder.joblib")
    lb_model = jb.load("./ml/model/lb.joblib")

    clean_data = data_preprocessing(data=data)

    predict_data, _, _, _ = process_data(clean_data,
                                         categorical_features=PROCESSED_CAT_FEATURES,
                                         lb=lb_model,
                                         training=False,
                                         label=None,
                                         encoder=encoder_model)
    predictions = inference(model, predict_data)
    original_output = lb_model.inverse_transform(predictions, threshold=None)
    print(original_output[0])
    return original_output[0]



