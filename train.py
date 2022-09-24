"""
Module that calls the training step.

Author: Bernardo C.
Date: 2022/09/21
"""
from train_model import train_model_implementation


def main(data_path=None):
    """
    Function that calls the train model

    Input:
        data_path: (str)
            Data path for the training step
    """
    train_model_implementation(data_path)


if __name__ == "__main__":
    main()
