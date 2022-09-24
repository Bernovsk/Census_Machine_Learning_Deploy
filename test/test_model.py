import logging
import os
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..', os.path.dirname('ml'))))
from inference_ import run_inference


logging.basicConfig(
    filename='./logs/test_train_data.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')




def test_slice_file():
    assert 'sliced_output.txt' in os.listdir('../ml/model/')


def test_inference(load_inference_data):

    result = run_inference(load_inference_data)
    assert result == "<=50K"
