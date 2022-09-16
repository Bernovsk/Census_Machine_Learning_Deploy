from ml.model_ import *
from ml.data import *
from ml.constants import *
#import argparse
from joblib import load

def main(data_path):
    model   = load("./ml/model/model.joblib")
    encoder = load("./ml/model/encoder.joblib")
    lb      = load("./ml/model/lb.joblib")
    



    test_data = data_preprocessing(data_path = data_path)

    check_slices(model = model, test_data = test_data, encoder = encoder, lb = lb)


if __name__ == '__main__':

    #parser = argparse.ArgumentParser(description= 'Test Path')
    #parser.add_argument('--test_data_path', type=str, help = 'Fist number', required = True)
    #args = parser.parse_args()

    main(data_path = './data/test_census_data.csv')
