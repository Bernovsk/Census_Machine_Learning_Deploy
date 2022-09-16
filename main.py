from fastapi import FastAPI
from inference_ import *
import pandas as pd
from pydantic import BaseModel
from typing import Literal
from inference_ import *


app = FastAPI()



class DataSchemma(BaseModel):


    age: int

    workclass : Literal['State_gov', 'Self_emp_not_inc', 'Private',
                        'Federal_gov', 'Local_gov', 'Self_emp_inc', 'Without_pay']
    fnlgt : int

    education: Literal['Bachelors', 'HS_grad', '11th', 'Masters', '9th', 'Some_college',
                     'Assoc_acdm', '7th_8th', 'Doctorate', 'Assoc_voc', 'Prof_school',
                      '5th_6th', '10th', 'Preschool', '12th', '1st_4th']

    education_num : int

    marital_status : Literal['Never_married',
                            'Married_civ_spouse', 'Divorced', 'Married_spouse_absent',
                            'Separated', 'Married_AF_spouse', 'Widowed']

    occupation: Literal['Adm_clerical',
                        'Exec_managerial', 'Handlers_cleaners', 'Prof_specialty', 'Other_service',
                         'Sales', 'Transport_moving', 'Farming_fishing', 'Machine_op_inspct', 'Tech_support',
                          'Craft_repair', 'Protective_serv', 'Armed_Forces', 'Priv_house_serv']

    relationship: Literal['Not_in_family', 'Husband', 'Wife',
                         'Own_child', 'Unmarried', 'Other_relative']

    race:  Literal['White', 'Black', 
                    'Asian_Pac_Islander', 'Amer_Indian_Eskimo',
                     'Other']

    sex : Literal['Male', 'Female']

    capital_gain : int

    capital_loss : int

    hours_per_week: int

    native_country : Literal['United_States',
                            'Cuba', 'Jamaica', 'India', 'Mexico', 'Puerto_Rico',
                            'Honduras', 'England', 'Canada', 'Germany',
                            'Iran', 'Philippines', 'Poland', 'Columbia',
                            'Cambodia', 'Thailand', 'Ecuador', 'Laos', 
                            'Taiwan', 'Haiti', 'Portugal', 'Dominican_Republic',
                            'El_Salvador', 'France', 'Guatemala', 'Italy', 'China',
                            'South', 'Japan', 'Yugoslavia', 'Peru', 'Outlying_US(Guam_USVI_etc)',
                            'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua',
                            'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand_Netherlands']


@app.get("/")
async def print_hello():
    return {"message":"Welcome to the model Api"}


@app.post("/inference")
async def inference(data : DataSchemma):
    
    dictionary_data = data.dict()
    keys_ = [key.replace('-', '_').strip() for key in dictionary_data.keys()]

    for new_key, old_key in zip(keys_, dictionary_data.keys()):
        dictionary_data[new_key] = dictionary_data[old_key]
        del dictionary_data[old_key]

    processed_input_data = pd.DataFrame(data = dictionary_data.values(), index = dictionary_data.keys()).transpose()

    prediction = run_inference(processed_input_data)
    
    return prediction
