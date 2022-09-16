from json import load
from ml.model_ import inference
from ml.data import *
from joblib import load
from ml.constants import *
import argparse

def run_inference(data):

    model   = load("./ml/model/model.joblib")
    encoder = load("./ml/model/encoder.joblib")
    lb      = load("./ml/model/lb.joblib")

    clean_data = data_preprocessing(data = data)

    X_data, _, _, _ = process_data(clean_data,
                                       categorical_features = PROCESSED_CAT_FEATURES,
                                       lb = lb,
                                       training= False,
                                       label = None,
                                       encoder = encoder)
    predictions = inference(model, X_data)
    original_output = lb.inverse_transform(predictions, threshold = None)                
    print(original_output[0])
    return original_output[0]
