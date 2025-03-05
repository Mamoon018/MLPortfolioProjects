
import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import datatransformationconfig
from src.components.data_transformation import Datatransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class Dataingestionconfig:
    train_data_path: str = os.path.join('artifacts','train.csv') #To store train.csv file in this path
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
class dataingestion:
    def __init__(self):
        self.ingestion_config = Dataingestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method to read the data")
        try:
            df = pd.read_csv('notebook\\data\\Failure.csv') 
            logging.info('imported and read the raw data')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
        
            logging.info('Train Test split initiated')

            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Data Ingestion is completed.')
        
            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    obj=dataingestion()
    obj.initiate_data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = Datatransformation()
    #data_transformation.initiate_data_transformation(train_data, test_data)
#while applying model

    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    modelTrainer = ModelTrainer()

    print(modelTrainer.initiating_model_trainer(train_arr, test_arr))
