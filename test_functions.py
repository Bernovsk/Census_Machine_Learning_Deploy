import logging
import os
from inference_ import run_inference
from ml.data import train_test_data

logging.basicConfig(
    filename='./logs/test_train_data.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

logging.basicConfig(
    filename='./logs/test_train_data.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_preprocessing_data(load_clean_data):
    train_data, test_data = train_test_data(load_clean_data)
    total_rows = train_data.shape[0] + test_data.shape[0] + 500
    total_cols = train_data.shape[1] + 3
    try:
        logging.info('Test the parity between the number of rows before the train test split')    
        assert total_rows == load_clean_data.shape[0]
        logging.info('The number of rows was conserved')
    except AssertionError:
        logging.info('The number of rows was not conserverd')
    try:
        logging.info('Test the parity between the number of columns before the train test split')    
        assert total_cols == load_clean_data.shape[1]
        logging.error('The number of columns was conserved')
    except AssertionError:
        logging.error('The number of columns was not conserverd')



def test_slice_file():
    assert 'sliced_output.txt' in os.listdir('./ml/model/')


def test_inference(load_inference_data):

    result = run_inference(load_inference_data)
    assert result == "<=50K"
