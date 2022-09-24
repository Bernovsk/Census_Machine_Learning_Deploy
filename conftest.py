# Bring your packages onto the path
import sys
import os
import pytest
import json
import joblib as jb
from fastapi.testclient import TestClient
import pandas as pd

sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname('main'), '..')))

from main import app


@pytest.fixture
def load_clean_data():
    clean_data = pd.read_csv('./data/pre_processing_census_data.csv',
                             sep=';')
    return clean_data


@pytest.fixture
def load_inference_data():
    dictionary_data = json.load(open('./example/inference_test.json'))
    if {type(val) for val in dictionary_data.values()} != {list}:
        for key, val in zip(dictionary_data.keys(), dictionary_data.values()):
            dictionary_data[key.replace('-', '_').strip()] = [val]
        processed_input_data = pd.DataFrame(dictionary_data)
    else:
        processed_input_data = pd.DataFrame(dictionary_data)

    return processed_input_data


@pytest.fixture
def load_model():
    model = jb.load('./ml/model/model.joblib')
    return model


@pytest.fixture
def load_lb():
    model = jb.load('./ml/model/lb.joblib')
    return model


@pytest.fixture
def load_encoder():
    model = jb.load('./ml/model/encoder.joblib')
    return model


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client
