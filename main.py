"""
Module that runs the local API and predict the target variable

Author: Bernardo C.
Date: 2022/09/21
"""
import os
from typing import Literal
from pydantic import BaseModel
import pandas as pd
from fastapi import FastAPI
from inference_ import run_inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        os.system.exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class DataSchemma(BaseModel):
    """
    Data Schemma class for the post method
    """
    age: int

    workclass: Literal['State_gov', 'Self_emp_not_inc',
                       'Private', 'Federal_gov',
                       'Local_gov', 'Self_emp_inc',
                       'Without_pay']
    fnlgt: int

    education: Literal['Bachelors', 'HS_grad', '11th',
                       'Masters', '9th', 'Some_college',
                       'Assoc_acdm', '7th_8th', 'Doctorate',
                       'Assoc_voc', 'Prof_school', '5th_6th',
                       '10th', 'Preschool', '12th', '1st_4th']

    education_num: int

    marital_status: Literal['Never_married', 'Married_civ_spouse',
                            'Divorced', 'Married_spouse_absent',
                            'Separated', 'Married_AF_spouse', 'Widowed']

    occupation: Literal['Adm_clerical', 'Exec_managerial',
                        'Handlers_cleaners', 'Prof_specialty',
                        'Other_service', 'Sales',
                        'Transport_moving', 'Farming_fishing',
                        'Machine_op_inspct', 'Tech_support',
                        'Craft_repair', 'Protective_serv',
                        'Armed_Forces', 'Priv_house_serv']

    relationship: Literal['Not_in_family', 'Husband', 'Wife',
                          'Own_child', 'Unmarried', 'Other_relative']

    race: Literal['White', 'Black',
                  'Asian_Pac_Islander', 'Amer_Indian_Eskimo',
                  'Other']

    sex: Literal['Male', 'Female']

    capital_gain: int

    capital_loss: int

    hours_per_week: int

    native_country: Literal['United_States', 'Cuba', 'Jamaica',
                            'India', 'Mexico', 'Puerto_Rico',
                            'Honduras', 'England', 'Canada',
                            'Germany', 'Iran', 'Philippines',
                            'Poland', 'Columbia', 'Cambodia',
                            'Thailand', 'Ecuador', 'Laos',
                            'Taiwan', 'Haiti', 'Portugal',
                            'Dominican_Republic', 'El_Salvador', 'France',
                            'Guatemala', 'Italy', 'China',
                            'South', 'Japan', 'Yugoslavia',
                            'Peru', 'Outlying_US(Guam_USVI_etc)', 'Scotland',
                            'Trinadad&Tobago', 'Greece', 'Nicaragua',
                            'Vietnam', 'Hong', 'Ireland',
                            'Hungary', 'Holand_Netherlands']


app = FastAPI()


@app.get("/")
async def print_hello():
    """
    Function that reproduce the get method in the FastAPI

    Input:
        None
    Output:
        (str)
    """
    return {"message": "Welcome to the model FastAPI homepage"}


@app.post("/")
async def inference_main(data: DataSchemma):
    """
    Function that reproduce the post method and return an salary prediction
    Input:
        data: (json or Dictionary)
            Dictionary that contains the DataSchema structure,
            following the correct keys data type with the
            possible values that the categorical variables can
            be present
    Output:
        (dictionary)
            The prediction from the post method.
    """
    dictionary_data = data.dict()

    if {type(val) for val in dictionary_data.values()} != {list}:
        for key, val in zip(dictionary_data.keys(), dictionary_data.values()):
            dictionary_data[key.replace('-', '_').strip()] = [val]
        processed_input_data = pd.DataFrame(dictionary_data)
    else:
        processed_input_data = pd.DataFrame(dictionary_data)

    prediction = run_inference(processed_input_data)

    return {"Salary_prediction": prediction}
