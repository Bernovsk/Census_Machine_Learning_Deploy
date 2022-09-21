import requests
import json

test_data = json.load(open('./example/inference_test.json'))
response = requests.post('https://census-final-prediction-app.herokuapp.com/', data=json.dumps(test_data))

print(f"Response status: {response.status_code}")
print(f"Salary Prediction: {response.json()['Salary_prediction']}")