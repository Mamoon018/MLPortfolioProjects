
import sys 
import os 
import pandas as pd 
from src.exception import CustomException
from src.utils import load_object

class predictpipeline:
    def __init__(self):
        pass 

    def predict(self,features):
        try:
            model_path = os.path.join('artifact', 'model.pkl')
            preprocessor_path = os.path.join('artifact', 'preprocessor.pkl')
            print('Before loading the files')

            model = load_object(file_path= model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            print('Files loaded')

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)
        
