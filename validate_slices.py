from json import encoder
from ml.model import *
from constants import *
from joblib import load

def run_slice_check(data_file:pd.DataFrame) -> None:

    model   = load(f"{MODEL_PATH}/model.joblib")
    encoder = load(f"{MODEL_PATH}/encoder.joblib")
    lb      = load(f"{MODEL_PATH}/lb.joblib")


    X_data, y_data, _, _ = process_data(data_file,
                    categorical_features = CATEGORICAL_FEATURES,
                    lb = lb,
                    encoder = encoder)
                    
    check_slices(model, X_data = X_data, y_data = y_data)