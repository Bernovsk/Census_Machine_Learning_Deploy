import json


def test_get_method(client):
    """
    Test the get method of the API
    """
    r = client.get('/')
    print(r)
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the model FastAPI homepage"}


def test_post_lower(client):
    """
    Test the prediction of data with true label <=50K
    """

    r = client.post('/',
                    json=json.dumps({
                                    "age": 50,
                                    "workclass": "Private",
                                    "fnlgt": 210008,
                                    "education": "HS_grad",
                                    "education_num": 9,
                                    "marital_status": "Never_married",
                                    "occupation": "Sales",
                                    "relationship": "Own_child",
                                    "race": "White",
                                    "sex": "Female",
                                    "capital_gain": 0,
                                    "capital_loss": 0,
                                    "hours_per_week": 40,
                                    "native_country": "United_States"
                                    }),
                    timeout=30)
    try:
        assert r.status_code == 200
    except AssertionError:
        print("Failed to post method {}".format(r.status_code))
    try:
        assert r.json() == {"Salary_prediction": "<=50K"}
    except AssertionError:
        print(r.json())


def test_post_wrong(client):
    """
    Test the prediction of data with true label <=50K
    """

    r = client.post('/',
                    json=json.dumps({
                        "age": 50,
                                    "workc": "--",
                                    "fnlgt": 210008,
                                    "educatin": "HS_grad",
                                    "educaton_num": [],
                                    "mrital_status": "Never_married",
                                    "occupat": "Sals",
                                    "relationsip": "Own_cild",
                                    "race": "White",
                                    "sex": "Female",
                                    "capital_gain": 0,
                                    "capital_loss": 0,
                                    "hours_per_week": 4,
                                    "native_cuntry": "United_Sttes"
                                    }),
                    timeout=30)

    assert r.status_code == 422


def test_post_higher(client):
    """
    Test the prediction of data with true label >50K
    """

    r = client.post('/',
                    data=json.dumps({
                                    "age": 30,
                                    "workclass": "Private",
                                    "fnlgt": 210008,
                                    "education": "HS_grad",
                                    "education_num": 9,
                                    "marital_status": "Never_married",
                                    "occupation": "Sales",
                                    "relationship": "Own_child",
                                    "race": "White",
                                    "sex": "male",
                                    "capital_gain": 60000,
                                    "capital_loss": 0,
                                    "hours_per_week": 40,
                                    "native_country": "United_States"
                                    }),
                    timeout=30)

    try:
        assert r.status_code == 200
    except AssertionError:
        print("Failed to post method {}".format(r.status_code))
    try:
        assert r.json() == {"Salary_prediction": ">50K"}
    except AssertionError:
        print(r.json())
