import sys
import os
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import Exception
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class Preprocessor:
    processing_path=os.path.join('resources','preprocessor.pkl')

class DataTransform:
    def __init__(self):
        self.processing_path=Preprocessor()

    def transformations(self):
        try:
            num_columns=["reading_score","writing_score"]
            categorical_columns=[ "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",]
            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )
            categorical_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))

                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {num_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_columns),
                ("cat_pipelines",categorical_pipeline,categorical_columns)

                ]
            )
            return preprocessor
        except Exception as e:
            raise Exception(e,sys)
        
    def initiate_Transformations(self,train_path,test_path):
        try:
            train=pd.read_csv(train_path)
            test=pd.read_csv(test_path)

            transformer=self.transformations()

            target_column_name="math_score"

            input_feature_train=train.drop(columns=[target_column_name],axis=1)
            target_feature_train=train[target_column_name]
            
            input_feature_test=test.drop(columns=[target_column_name],axis=1)
            target_feature_test=test[target_column_name]

            input_feature_train_transform=transformer.fit_transform(input_feature_train)
            input_feature_test_transform=transformer.transform(input_feature_test)



            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            train_arr = np.c_[
                input_feature_train_transform, np.array(target_feature_train)
            ]
            test_arr = np.c_[input_feature_test_transform, np.array(target_feature_test)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.processing_path.processing_path,
                obj=transformer

            )

            return (
                train_arr,
                test_arr,
                self.processing_path.processing_path
            )


        except Exception as e:
            raise Exception(e,sys)


          