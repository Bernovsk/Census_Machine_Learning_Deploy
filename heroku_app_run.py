"""
Module that runs test the
connection and post method on Heroku API

Author: Bernardo C.
Date: 2022/09/21
"""

from requests import post

with open('./example/inference_test.json', encoding='UTF-8') as test_data:

    response = post('https://census-final-prediction-app.herokuapp.com/',
                    data=test_data,
                    timeout=30)

    print(f"Response status: {response.status_code}")
    print(f"Salary Prediction: {response.json()['Salary_prediction']}")
