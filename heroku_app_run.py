"""
Module that runs test the
connection and post method on Heroku API

Author: Bernardo C.
Date: 2022/09/21
"""

import json
import requests

with json.load(open('./example/inference_test.json', encoding='UTF-8')) as data:
    test_data = data


response = requests.post('https://census-final-prediction-app.herokuapp.com/',
                         data=json.dumps(test_data),
                         timeout=30)

print(f"Response status: {response.status_code}")
print(f"Salary Prediction: {response.json()['Salary_prediction']}")
