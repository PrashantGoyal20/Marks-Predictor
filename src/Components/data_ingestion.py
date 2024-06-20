import os 
import sys
from src.exception import Exception
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
from src.Components.data_transform import Preprocessor
from src.Components.data_transform import DataTransform
from src.Components.training import ModelTrainer

@dataclass
class Data_Ingestion_Config():
    train_path :str=os.path.join('resources', "train.csv")
    test_path :str=os.path.join('resources', "test.csv")
    raw_path :str=os.path.join('resources', "raw.csv")


class Data_Ingestion():
    def __init__(self):
        self.data_path = Data_Ingestion_Config()

    def data_ingestion_initiator(self):
        
        logging.info('Starting data injection')
        try:
            df=pd.read_csv('data\StudentsPerformance.csv')
            logging.info('Reading data')
            os.makedirs(os.path.dirname(self.data_path.test_path),exist_ok=True)
            df.to_csv(self.data_path.raw_path,index=False,header=True)

            logging.info('Train test split')
            train,test=train_test_split(df,test_size=0.2,random_state=42)
            train.to_csv(self.data_path.train_path,index=False,header=True)
            test.to_csv(self.data_path.test_path,index=False,header=True)

            logging.info('Data injection completed')

            return(
                self.data_path.train_path,
                self.data_path.test_path
            )

        except Exception as e:
            raise Exception(e)
        
__name__ = ('__manin__')        
obj=Data_Ingestion()
train,test=obj.data_ingestion_initiator()

data_transform=DataTransform()
train_array,test_array,pre_processing_path=data_transform.initiate_Transformations(train,test)

model=ModelTrainer()
model.initiate_training(train_array,test_array)