from json import load
from ml.model import inference
from ml.data import *
from joblib import load
from constants import *

def run_inference(data):

    model   = load(f"{MODEL_PATH}/model.joblib")
    encoder = load(f"{MODEL_PATH}/encoder.joblib")
    lb      = load(f"{MODEL_PATH}/lb.joblib")

    X_data, y_data, _, _ = process_data(data,
                                categorical_features = CATEGORICAL_FEATURES,
                                lb = lb,
                                encoder = encoder)
    predictions = inference(model, X_data)                

    return predictions