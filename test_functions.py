import os
from inference_ import run_inference
from ml.data import train_test_data


def test_preprocessing_data(load_clean_data):
    clean_data = load_clean_data
    train_data, test_data = train_test_data(clean_data)
    total_rows = train_data.shape[0] + test_data.shape[0]

    assert total_rows == clean_data.shape[0]
    assert train_data.shape[1] == clean_data.shape[1]


def test_slice_file():
    assert 'sliced_output.txt' in os.listdir('ml/model/')


def test_inference(load_inference_data):

    result = run_inference(load_inference_data)
    assert result == "<=50K"
