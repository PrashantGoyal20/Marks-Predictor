import sys
import pandas as pd
from src.exception import Exception
from src.utils import load_object


class Prediction:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path='./resources/preprocessor.pkl'
            model_path='./resources/model.pkl'
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            scaled_data=preprocessor.transform(features)
            prediction=model.predict(scaled_data)
            return(prediction)  
        except Exception as e:
            raise Exception(e,sys)  
        

class CustomData:
    def __init__(self,gender:str,race_ethnicity:str,parental_level_of_education:str,lunch:str,test_preparation_course:str,reading_score:int,writing_score:int):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score  


    def convert_to_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise Exception(e,sys)
