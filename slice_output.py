"""
Module that runs the validation over
the slice of each categorical column value.

Author: Bernardo C.
Date: 2022/09/21
"""
from joblib import load
from ml.model_ import check_slices
from ml.data import data_preprocessing


def main(data_path):
    """
    Validate the accuracy of the model
    in each slice of the categorical variables

    Input:
        data_path: (str)
            Test data path
    Output:
        sliced_output.txt in the ./ml/model Folder
    """
    model = load("./ml/model/model.joblib")
    encoder_model = load("./ml/model/encoder.joblib")
    lb_model = load("./ml/model/lb.joblib")

    test_data = data_preprocessing(data_path)

    check_slices(
        model=model,
        test_data=test_data,
        encoder_model=encoder_model,
        lb_model=lb_model)


if __name__ == '__main__':

    main(data_path='./data/test_census_data.csv')
